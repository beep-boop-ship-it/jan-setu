package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/crypto/bcrypt"
)

type Report struct {
	Title       string `json:"title"`
	Category    string `json:"category"`
	District    string `json:"district"`
	Location    string `json:"location"`
	PinCode     string `json:"pin_code"`
	Description string `json:"description"`
}

type SignupRequest struct {
	FullName     string `json:"full_name"`
	Institution  string `json:"institution"`
	EnrollmentID string `json:"enrollment_id"`
	Email        string `json:"email"`
	Phone        string `json:"phone"`
	Password     string `json:"password"`
	Role         string `json:"role"`
}

type SolverProfileRequest struct {
	FullName     string `json:"full_name"`
	Phone        string `json:"phone"`
	University   string `json:"university"`
	EnrollmentID string `json:"enrollment_id"`
	Department   string `json:"department"`
	YearOfStudy  string `json:"year_of_study"`
	Skills       string `json:"skills"`
	Interests    string `json:"interests"`
	Projects     string `json:"projects"`
}

func generateTrackID() (string, error) {
	b := make([]byte, 8)

	if _, err := rand.Read(b); err != nil {
		return "", err
	}

	return "JS-2026-" + strings.ToUpper(hex.EncodeToString(b)), nil
}

func getSupabaseConfig() (string, string, error) {
	supabaseURL := strings.TrimRight(os.Getenv("SUPABASE_URL"), "/")

	supabaseKey := os.Getenv("SUPABASE_SERVICE_ROLE_KEY")

	if supabaseKey == "" {
		supabaseKey = os.Getenv("SUPABASE_SECRET_KEY")
	}

	if supabaseURL == "" || supabaseKey == "" {
		return "", "", fmt.Errorf("Supabase environment variables are missing")
	}

	return supabaseURL, supabaseKey, nil
}

func uploadReportPhoto(
	file multipart.File,
	header *multipart.FileHeader,
	reportID int64,
	photoNumber int,
) (string, error) {

	supabaseURL, supabaseKey, err := getSupabaseConfig()
	if err != nil {
		return "", err
	}

	const maxPhotoSize = 6 * 1024 * 1024

	if header.Size > maxPhotoSize {
		return "", fmt.Errorf("photo %d exceeds 6 MB", photoNumber)
	}

	ext := strings.ToLower(filepath.Ext(header.Filename))

	switch ext {
	case ".jpg", ".jpeg", ".png", ".webp":
	default:
		return "", fmt.Errorf("photo %d has an unsupported file type", photoNumber)
	}

	data, err := io.ReadAll(io.LimitReader(file, maxPhotoSize+1))
	if err != nil {
		return "", err
	}

	if len(data) > maxPhotoSize {
		return "", fmt.Errorf("photo %d exceeds 6 MB", photoNumber)
	}

	contentType := header.Header.Get("Content-Type")

	if contentType == "" {
		switch ext {
		case ".jpg", ".jpeg":
			contentType = "image/jpeg"
		case ".png":
			contentType = "image/png"
		case ".webp":
			contentType = "image/webp"
		}
	}

	objectPath := fmt.Sprintf(
		"report-%d/photo-%d%s",
		reportID,
		photoNumber,
		ext,
	)

	uploadURL := fmt.Sprintf(
		"%s/storage/v1/object/report-photos/%s",
		supabaseURL,
		objectPath,
	)

	req, err := http.NewRequest(
		http.MethodPost,
		uploadURL,
		bytes.NewReader(data),
	)
	if err != nil {
		return "", err
	}

	req.Header.Set("Authorization", "Bearer "+supabaseKey)
	req.Header.Set("apikey", supabaseKey)
	req.Header.Set("Content-Type", contentType)
	req.Header.Set("x-upsert", "false")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)

		return "", fmt.Errorf(
			"Supabase Storage upload failed: %s",
			strings.TrimSpace(string(body)),
		)
	}

	publicURL := fmt.Sprintf(
		"%s/storage/v1/object/public/report-photos/%s",
		supabaseURL,
		objectPath,
	)

	return publicURL, nil
}

func signup(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req SignupRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.FullName == "" ||
			req.Institution == "" ||
			req.EnrollmentID == "" ||
			req.Email == "" ||
			req.Phone == "" ||
			req.Password == "" {
			http.Error(w, "All required fields must be provided", http.StatusBadRequest)
			return
		}

		if req.Role == "" {
			req.Role = "student"
		}

		passwordHash, err := bcrypt.GenerateFromPassword(
			[]byte(req.Password),
			bcrypt.DefaultCost,
		)

		if err != nil {
			http.Error(w, "Failed to secure password", http.StatusInternalServerError)
			return
		}

		var id int64

		err = db.QueryRow(
			context.Background(),
			`
			INSERT INTO solver_accounts
				(full_name, institution, enrollment_id, email, phone, password_hash, role)
			VALUES
				($1, $2, $3, $4, $5, $6, $7)
			RETURNING id
			`,
			req.FullName,
			req.Institution,
			req.EnrollmentID,
			req.Email,
			req.Phone,
			string(passwordHash),
			req.Role,
		).Scan(&id)

		if err != nil {
			http.Error(w, "Failed to create account", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(map[string]any{
			"id":      id,
			"message": "Account created successfully",
		})
	}
}

func createReport(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var report Report
		var photos []*multipart.FileHeader

		contentType := r.Header.Get("Content-Type")

		// JSON support for existing clients
		if strings.HasPrefix(contentType, "application/json") {

			if err := json.NewDecoder(r.Body).Decode(&report); err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}

			// Multipart support for report photos
		} else if strings.HasPrefix(contentType, "multipart/form-data") {

			err := r.ParseMultipartForm(40 << 20)
			if err != nil {
				http.Error(w, "Invalid multipart form", http.StatusBadRequest)
				return
			}

			report.Title = r.FormValue("title")
			report.Category = r.FormValue("category")
			report.District = r.FormValue("district")
			report.Location = r.FormValue("location")
			report.PinCode = r.FormValue("pin_code")
			report.Description = r.FormValue("description")

			for i := 1; i <= 5; i++ {
				fieldName := fmt.Sprintf("photo%d", i)

				_, header, err := r.FormFile(fieldName)

				if err != nil {
					if err == http.ErrMissingFile {
						continue
					}

					http.Error(
						w,
						fmt.Sprintf("Failed to read %s", fieldName),
						http.StatusBadRequest,
					)
					return
				}

				photos = append(photos, header)
			}

		} else {
			http.Error(w, "Unsupported Content-Type", http.StatusBadRequest)
			return
		}

		if strings.TrimSpace(report.Title) == "" ||
			strings.TrimSpace(report.Category) == "" ||
			strings.TrimSpace(report.District) == "" ||
			strings.TrimSpace(report.Location) == "" ||
			strings.TrimSpace(report.PinCode) == "" ||
			strings.TrimSpace(report.Description) == "" {

			http.Error(
				w,
				"All required report fields must be provided",
				http.StatusBadRequest,
			)
			return
		}

		trackID, err := generateTrackID()
		if err != nil {
			log.Println("Track ID generation error:", err)

			http.Error(
				w,
				"Failed to generate Track ID",
				http.StatusInternalServerError,
			)
			return
		}

		var id int64

		err = db.QueryRow(
			context.Background(),
			`
			INSERT INTO reports
				(
					title,
					category,
					district,
					location,
					pin_code,
					description,
					track_id
				)
			VALUES
				($1, $2, $3, $4, $5, $6, $7)
			RETURNING id
			`,
			report.Title,
			report.Category,
			report.District,
			report.Location,
			report.PinCode,
			report.Description,
			trackID,
		).Scan(&id)

		if err != nil {
			log.Println("Report creation error:", err)

			http.Error(
				w,
				"Failed to create report",
				http.StatusInternalServerError,
			)
			return
		}

		photoURLs := []string{}

		for i, header := range photos {

			file, err := header.Open()
			if err != nil {
				log.Println("Photo open error:", err)

				http.Error(
					w,
					"Failed to open uploaded photo",
					http.StatusInternalServerError,
				)
				return
			}

			url, err := uploadReportPhoto(
				file,
				header,
				id,
				i+1,
			)

			file.Close()

			if err != nil {
				log.Println("Photo upload error:", err)

				http.Error(
					w,
					"Failed to upload report photo",
					http.StatusInternalServerError,
				)
				return
			}

			photoURLs = append(photoURLs, url)
		}

		if len(photoURLs) > 0 {

			photoJSON, err := json.Marshal(photoURLs)
			if err != nil {
				http.Error(
					w,
					"Failed to process photo URLs",
					http.StatusInternalServerError,
				)
				return
			}

			_, err = db.Exec(
				context.Background(),
				`
				UPDATE reports
				SET photo_urls = $1
				WHERE id = $2
				`,
				photoJSON,
				id,
			)

			if err != nil {
				log.Println("Photo URL database error:", err)

				http.Error(
					w,
					"Failed to save photo information",
					http.StatusInternalServerError,
				)
				return
			}
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(map[string]any{
			"id":         id,
			"track_id":   trackID,
			"status":     "submitted",
			"photo_urls": photoURLs,
		})
	}
}

func verifyTrackID(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			ReportID int64  `json:"report_id"`
			TrackID  string `json:"track_id"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.ReportID <= 0 || strings.TrimSpace(req.TrackID) == "" {
			http.Error(
				w,
				"Report ID and Track ID are required",
				http.StatusBadRequest,
			)
			return
		}

		type ReportResponse struct {
			ID          int64           `json:"id"`
			Title       string          `json:"title"`
			Category    string          `json:"category"`
			District    string          `json:"district"`
			Location    string          `json:"location"`
			PinCode     string          `json:"pin_code"`
			Description string          `json:"description"`
			Status      string          `json:"status"`
			CreatedAt   time.Time       `json:"created_at"`
			PhotoURLs   json.RawMessage `json:"photo_urls"`
		}

		var report ReportResponse

		err := db.QueryRow(
			context.Background(),
			`
			SELECT id, title, category, district, location,
       pin_code, description, status, created_at, photo_urls
FROM public.reports
			WHERE id = $1
			  AND track_id = $2
			`,
			req.ReportID,
			strings.TrimSpace(req.TrackID),
		).Scan(
			&report.ID,
			&report.Title,
			&report.Category,
			&report.District,
			&report.Location,
			&report.PinCode,
			&report.Description,
			&report.Status,
			&report.CreatedAt,
			&report.PhotoURLs,
		)

		if err != nil {

			if err == pgx.ErrNoRows {
				http.Error(
					w,
					"Track ID does not match this problem",
					http.StatusUnauthorized,
				)
				return
			}

			log.Println("Track ID verification error:", err)

			http.Error(
				w,
				"Failed to verify Track ID",
				http.StatusInternalServerError,
			)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(report)
	}
}

func getReports(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		rows, err := db.Query(
			context.Background(),
			`SELECT id, title, category, district, location,
            pin_code, description, status, created_at, photo_urls
     FROM public.reports
     ORDER BY created_at DESC`,
		)

		if err != nil {
			log.Println("Report query error:", err)
			http.Error(w, "Failed to fetch reports", http.StatusInternalServerError)
			return
		}

		defer rows.Close()

		type ReportResponse struct {
			ID          int64     `json:"id"`
			Title       string    `json:"title"`
			Category    string    `json:"category"`
			District    string    `json:"district"`
			Location    string    `json:"location"`
			PinCode     string    `json:"pin_code"`
			Description string    `json:"description"`
			Status      string    `json:"status"`
			CreatedAt   time.Time `json:"created_at"`
			PhotoURLs   []string  `json:"photo_urls"`
		}

		reports := []ReportResponse{}

		for rows.Next() {

			var report ReportResponse

			err := rows.Scan(
				&report.ID,
				&report.Title,
				&report.Category,
				&report.District,
				&report.Location,
				&report.PinCode,
				&report.Description,
				&report.Status,
				&report.CreatedAt,
				&report.PhotoURLs,
			)

			if err != nil {
				log.Println("Report scan error:", err)
				http.Error(w, "Failed to read reports", http.StatusInternalServerError)
				return
			}

			reports = append(reports, report)
		}

		if err := rows.Err(); err != nil {
			http.Error(w, "Failed to read reports", http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(reports)
	}
}

func getReportByID(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		type ReportResponse struct {
			ID          int64     `json:"id"`
			Title       string    `json:"title"`
			Category    string    `json:"category"`
			District    string    `json:"district"`
			Location    string    `json:"location"`
			PinCode     string    `json:"pin_code"`
			Description string    `json:"description"`
			Status      string    `json:"status"`
			CreatedAt   time.Time `json:"created_at"`
		}

		var report ReportResponse

		err := db.QueryRow(
			context.Background(),
			`
			SELECT id, title, category, district, location,
			       pin_code, description, status, created_at
			FROM public.reports
			WHERE id = $1
			`,
			r.PathValue("id"),
		).Scan(
			&report.ID,
			&report.Title,
			&report.Category,
			&report.District,
			&report.Location,
			&report.PinCode,
			&report.Description,
			&report.Status,
			&report.CreatedAt,
		)

		if err != nil {
			http.Error(w, "Report not found", http.StatusNotFound)
			return
		}

		json.NewEncoder(w).Encode(report)
	}
}

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func login(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req LoginRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.Email == "" || req.Password == "" {
			http.Error(w, "Email and password are required", http.StatusBadRequest)
			return
		}

		var (
			id           int64
			fullName     string
			role         string
			passwordHash string
		)

		err := db.QueryRow(
			context.Background(),
			`
			SELECT id, full_name, role, password_hash
			FROM solver_accounts
			WHERE email = $1
			`,
			req.Email,
		).Scan(
			&id,
			&fullName,
			&role,
			&passwordHash,
		)

		if err != nil {
			http.Error(w, "Invalid email or password", http.StatusUnauthorized)
			return
		}

		if err := bcrypt.CompareHashAndPassword(
			[]byte(passwordHash),
			[]byte(req.Password),
		); err != nil {
			http.Error(w, "Invalid email or password", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(map[string]any{
			"id":        id,
			"full_name": fullName,
			"role":      role,
			"message":   "Login successful",
		})
	}
}

func getSolverProfile(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		accountIDStr := strings.TrimPrefix(
			r.URL.Path,
			"/api/solver/profile/",
		)

		accountID, err := strconv.ParseInt(accountIDStr, 10, 64)
		if err != nil {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		var profile SolverProfileRequest

		err = db.QueryRow(
			context.Background(),
			`
			SELECT
				full_name,
				phone,
				university,
				enrollment_id,
				department,
				year_of_study,
				skills,
				interests,
				projects
			FROM solver_profiles
			WHERE account_id = $1
			`,
			accountID,
		).Scan(
			&profile.FullName,
			&profile.Phone,
			&profile.University,
			&profile.EnrollmentID,
			&profile.Department,
			&profile.YearOfStudy,
			&profile.Skills,
			&profile.Interests,
			&profile.Projects,
		)

		if err != nil {
			if err == pgx.ErrNoRows {
				http.Error(w, "Profile not found", http.StatusNotFound)
				return
			}

			log.Println("Profile lookup error:", err)
			http.Error(w, "Failed to fetch profile", http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(profile)
	}
}

func saveSolverProfile(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, PUT, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost &&
			r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		accountIDStr := strings.TrimPrefix(
			r.URL.Path,
			"/api/solver/profile/",
		)

		accountID, err := strconv.ParseInt(accountIDStr, 10, 64)
		if err != nil {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		var req SolverProfileRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		_, err = db.Exec(
			context.Background(),
			`
			INSERT INTO solver_profiles (
				account_id,
				full_name,
				phone,
				university,
				enrollment_id,
				department,
				year_of_study,
				skills,
				interests,
				projects
			)
			VALUES (
				$1,$2,$3,$4,$5,$6,$7,$8,$9,$10
			)
			ON CONFLICT (account_id)
			DO UPDATE SET
				full_name = EXCLUDED.full_name,
				phone = EXCLUDED.phone,
				university = EXCLUDED.university,
				enrollment_id = EXCLUDED.enrollment_id,
				department = EXCLUDED.department,
				year_of_study = EXCLUDED.year_of_study,
				skills = EXCLUDED.skills,
				interests = EXCLUDED.interests,
				projects = EXCLUDED.projects,
				updated_at = NOW()
			`,
			accountID,
			req.FullName,
			req.Phone,
			req.University,
			req.EnrollmentID,
			req.Department,
			req.YearOfStudy,
			req.Skills,
			req.Interests,
			req.Projects,
		)

		if err != nil {
			log.Println("Profile save error:", err)
			http.Error(w, "Failed to save profile", http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)

		json.NewEncoder(w).Encode(map[string]any{
			"message": "Profile saved successfully",
		})
	}
}

func solverProfile(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		switch r.Method {

		case http.MethodGet:
			getSolverProfile(db)(w, r)

		case http.MethodPost,
			http.MethodPut,
			http.MethodOptions:
			saveSolverProfile(db)(w, r)

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}
