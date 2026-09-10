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

// ============================================================
// INDUSTRY AI MATCHING
// ============================================================

type IndustryOrganization struct {
	ID                   string   `json:"id"`
	Name                 string   `json:"name"`
	OrganizationType     string   `json:"organization_type"`
	Description          string   `json:"description"`
	Domains              []string `json:"domains"`
	Skills               []string `json:"skills"`
	ResearchAreas        []string `json:"research_areas"`
	Facilities           []string `json:"facilities"`
	AvailableCapacity    int      `json:"available_capacity"`
	IndustryCapabilities []string `json:"industry_capabilities"`
	Verified             bool     `json:"verified"`
}

type IndustryProblem struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Category    string `json:"category"`
	District    string `json:"district"`
	Location    string `json:"location"`
	PinCode     string `json:"pin_code"`
}

type IndustryMatchRequest struct {
	Organization IndustryOrganization `json:"organization"`
	Problems     []IndustryProblem    `json:"problems"`
}

type IndustryMatchResponse struct {
	Matches []map[string]interface{} `json:"matches"`
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

// ============================================================
// AI INTEGRATION
// ============================================================

type ExistingAIProblem struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Domain      string   `json:"domain"`
	Tags        []string `json:"tags"`
	Locality    string   `json:"locality"`
}

type AIReportRequest struct {
	Title            string              `json:"title"`
	Description      string              `json:"description"`
	Locality         string              `json:"locality"`
	District         string              `json:"district"`
	PinCode          string              `json:"pin_code"`
	ExistingProblems []ExistingAIProblem `json:"existing_problems"`
}

type AIReportResponse struct {
	Status           string           `json:"status"`
	Decision         string           `json:"decision"`
	DecisionReasons  []string         `json:"decision_reasons"`
	Analysis         AIAnalysis       `json:"analysis"`
	ProblemStatement map[string]any   `json:"problem_statement"`
	DuplicateMatches []map[string]any `json:"duplicate_matches"`
}

type AIAnalysis struct {
	Category string `json:"category"`
	Domain   string `json:"domain"`
}

// ============================================================
// FETCH EXISTING REPORTS FOR AI DEDUPLICATION
// ============================================================

func getExistingProblemsForAI(
	db *pgxpool.Pool,
) ([]ExistingAIProblem, error) {

	rows, err := db.Query(
		context.Background(),
		`
		SELECT
			id,
			title,
			description,
			category,
			location
		FROM public.reports
		WHERE status NOT IN ('REJECTED', 'INVALID')
		ORDER BY created_at DESC
		LIMIT 1000
		`,
	)

	if err != nil {
		return nil, fmt.Errorf(
			"failed to fetch existing reports for deduplication: %w",
			err,
		)
	}

	defer rows.Close()

	problems := make([]ExistingAIProblem, 0)

	for rows.Next() {
		var (
			id          int64
			title       string
			description string
			category    string
			location    string
		)

		if err := rows.Scan(
			&id,
			&title,
			&description,
			&category,
			&location,
		); err != nil {
			return nil, fmt.Errorf(
				"failed to read existing report for deduplication: %w",
				err,
			)
		}

		problems = append(problems, ExistingAIProblem{
			ID:          strconv.FormatInt(id, 10),
			Title:       title,
			Description: description,
			Domain:      "",
			Tags:        []string{},
			Locality:    location,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf(
			"failed while reading existing reports: %w",
			err,
		)
	}

	return problems, nil
}

// ============================================================
// SEND REPORT + EXISTING REPORTS TO PYTHON AI
// ============================================================

func analyzeReportWithAI(
	db *pgxpool.Pool,
	report Report,
) (*AIReportResponse, error) {

	aiURL := strings.TrimRight(
		os.Getenv("JANSETU_AI_URL"),
		"/",
	)

	if aiURL == "" {
		aiURL = "http://localhost:8000"
	}

	// ----------------------------------------------------------
	// Fetch existing reports from PostgreSQL.
	//
	// Go owns database access.
	// Python never connects directly to Supabase.
	// ----------------------------------------------------------

	existingProblems, err := getExistingProblemsForAI(db)
	if err != nil {
		return nil, err
	}

	log.Printf(
		"Loaded %d existing reports for AI deduplication",
		len(existingProblems),
	)

	requestBody := AIReportRequest{
		Title:            strings.TrimSpace(report.Title),
		Description:      strings.TrimSpace(report.Description),
		Locality:         strings.TrimSpace(report.Location),
		District:         strings.TrimSpace(report.District),
		PinCode:          strings.TrimSpace(report.PinCode),
		ExistingProblems: existingProblems,
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to encode AI request: %w",
			err,
		)
	}

	req, err := http.NewRequest(
		http.MethodPost,
		aiURL+"/analyze-report",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to create AI request: %w",
			err,
		)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf(
			"AI service unavailable: %w",
			err,
		)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to read AI response: %w",
			err,
		)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf(
			"AI service returned HTTP %d: %s",
			resp.StatusCode,
			strings.TrimSpace(string(responseBody)),
		)
	}

	var result AIReportResponse

	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, fmt.Errorf(
			"invalid AI response: %w",
			err,
		)
	}

	log.Printf(
		"AI RESULT: status=%s decision=%s matches=%d",
		result.Status,
		result.Decision,
		len(result.DuplicateMatches),
	)

	for i, match := range result.DuplicateMatches {
		log.Printf("AI MATCH %d: %+v", i+1, match)
	}

	return &result, nil
}

// ============================================================
// SEND INDUSTRY PROFILE + CIVIC PROBLEMS TO PYTHON AI
// ============================================================

func truncateForAI(value string, max int) string {
	value = strings.TrimSpace(value)
	if len(value) <= max {
		return value
	}
	return value[:max]
}

func matchIndustryWithAI(
	db *pgxpool.Pool,
	accountID int64,
) (*IndustryMatchResponse, error) {

	aiURL := strings.TrimRight(
		os.Getenv("JANSETU_AI_URL"),
		"/",
	)

	if aiURL == "" {
		aiURL = "http://localhost:8000"
	}

	// ----------------------------------------------------------
	// Fetch company account + profile
	// ----------------------------------------------------------

	var organization IndustryOrganization

	var fullName string
	var institution string
	var phone string
	var university string
	var department string
	var skills string
	var interests string
	var projects string

	err := db.QueryRow(
		context.Background(),
		`
		SELECT
			COALESCE(sa.full_name, ''),
			COALESCE(sa.institution, ''),
			COALESCE(sa.phone, ''),
			COALESCE(sp.university, ''),
			COALESCE(sp.department, ''),
			COALESCE(sp.skills, ''),
			COALESCE(sp.interests, ''),
			COALESCE(sp.projects, '')
		FROM solver_accounts sa
		LEFT JOIN solver_profiles sp
			ON sp.account_id = sa.id
		WHERE sa.id = $1
		  AND sa.role = 'company'
		`,
		accountID,
	).Scan(
		&fullName,
		&institution,
		&phone,
		&university,
		&department,
		&skills,
		&interests,
		&projects,
	)

	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, fmt.Errorf("company account/profile not found")
		}

		return nil, fmt.Errorf(
			"failed to fetch company profile: %w",
			err,
		)
	}

	// ----------------------------------------------------------
	// Convert existing profile fields into AI capabilities
	// ----------------------------------------------------------

	organization = IndustryOrganization{
		ID:               strconv.FormatInt(accountID, 10),
		Name:             strings.TrimSpace(fullName),
		OrganizationType: "industry",

		Description: truncateForAI(
			strings.TrimSpace(
				strings.Join(
					[]string{
						institution,
						university,
						department,
						interests,
						projects,
					},
					" ",
				),
			),
			1000,
		),

		Domains: []string{
			truncateForAI(department, 200),
		},

		Skills: []string{
			truncateForAI(skills, 200),
		},

		ResearchAreas: []string{
			truncateForAI(interests, 200),
		},

		Facilities: []string{},

		AvailableCapacity: 2,

		IndustryCapabilities: []string{
			truncateForAI(skills, 200),
			truncateForAI(interests, 200),
			truncateForAI(projects, 200),
		},

		Verified: true,
	}

	// ----------------------------------------------------------
	// Fetch civic problems that can currently be matched
	// ----------------------------------------------------------

	rows, err := db.Query(
		context.Background(),
		`
		SELECT
			id,
			COALESCE(title, ''),
			COALESCE(description, ''),
			COALESCE(category, ''),
			COALESCE(district, ''),
			COALESCE(location, ''),
			COALESCE(pin_code, '')
		FROM reports
		WHERE workflow_status <> 'SOLVED'
		ORDER BY created_at DESC
		`,
	)

	if err != nil {
		return nil, fmt.Errorf(
			"failed to fetch civic problems: %w",
			err,
		)
	}

	defer rows.Close()

	problems := make([]IndustryProblem, 0)

	for rows.Next() {

		var problem IndustryProblem

		var id int64

		err := rows.Scan(
			&id,
			&problem.Title,
			&problem.Description,
			&problem.Category,
			&problem.District,
			&problem.Location,
			&problem.PinCode,
		)

		if err != nil {
			return nil, fmt.Errorf(
				"failed to read civic problem: %w",
				err,
			)
		}

		problem.ID = strconv.FormatInt(id, 10)

		if len(strings.TrimSpace(problem.Description)) < 3 {
			continue
		}

		problems = append(
			problems,
			problem,
		)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf(
			"failed while reading civic problems: %w",
			err,
		)
	}

	log.Printf(
		"Industry AI: account=%d problems=%d",
		accountID,
		len(problems),
	)

	// ----------------------------------------------------------
	// Build Python AI request
	// ----------------------------------------------------------

	requestBody := IndustryMatchRequest{
		Organization: organization,
		Problems:     problems,
	}

	// log.Printf("Industry AI REQUEST: %+v", requestBody)

	body, err := json.Marshal(requestBody)

	if err != nil {
		return nil, fmt.Errorf(
			"failed to encode industry AI request: %w",
			err,
		)
	}

	// ----------------------------------------------------------
	// Call Python AI
	// ----------------------------------------------------------

	req, err := http.NewRequest(
		http.MethodPost,
		aiURL+"/match-industry",
		bytes.NewReader(body),
	)

	if err != nil {
		return nil, fmt.Errorf(
			"failed to create industry AI request: %w",
			err,
		)
	}

	req.Header.Set(
		"Content-Type",
		"application/json",
	)

	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	resp, err := client.Do(req)

	if err != nil {
		return nil, fmt.Errorf(
			"industry AI service unavailable: %w",
			err,
		)
	}

	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)

	if err != nil {
		return nil, fmt.Errorf(
			"failed to read industry AI response: %w",
			err,
		)
	}

	if resp.StatusCode < 200 ||
		resp.StatusCode >= 300 {

		return nil, fmt.Errorf(
			"industry AI returned HTTP %d: %s",
			resp.StatusCode,
			strings.TrimSpace(
				string(responseBody),
			),
		)
	}

	var result IndustryMatchResponse

	if err := json.Unmarshal(
		responseBody,
		&result,
	); err != nil {

		return nil, fmt.Errorf(
			"invalid industry AI response: %w",
			err,
		)
	}

	log.Printf(
		"Industry AI RESULT: account=%d matches=%d",
		accountID,
		len(result.Matches),
	)

	return &result, nil
}

// ============================================================
// GET INDUSTRY AI MATCHES
// ============================================================

func getIndustryMatches(db *pgxpool.Pool) http.HandlerFunc {

	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Access-Control-Allow-Methods",
			"GET, OPTIONS",
		)

		w.Header().Set(
			"Access-Control-Allow-Headers",
			"Content-Type",
		)

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodGet {
			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		// ------------------------------------------------------
		// /api/industry/matches/{accountID}
		// ------------------------------------------------------

		accountIDStr := strings.TrimPrefix(
			r.URL.Path,
			"/api/industry/matches/",
		)

		accountID, err := strconv.ParseInt(
			accountIDStr,
			10,
			64,
		)

		if err != nil || accountID <= 0 {
			http.Error(
				w,
				"Invalid account ID",
				http.StatusBadRequest,
			)
			return
		}

		result, err := matchIndustryWithAI(
			db,
			accountID,
		)

		if err != nil {

			log.Println(
				"Industry matching error:",
				err,
			)

			http.Error(
				w,
				err.Error(),
				http.StatusInternalServerError,
			)

			return
		}

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		json.NewEncoder(w).Encode(result)
	}
}

// ============================================================
// CREATE REPORT
// ============================================================

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

		if strings.HasPrefix(contentType, "application/json") {

			if err := json.NewDecoder(r.Body).Decode(&report); err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}

			// Category is AI-derived, never client-controlled.
			report.Category = ""

		} else if strings.HasPrefix(contentType, "multipart/form-data") {

			if err := r.ParseMultipartForm(40 << 20); err != nil {
				http.Error(w, "Invalid multipart form", http.StatusBadRequest)
				return
			}

			report.Title = r.FormValue("title")
			report.District = r.FormValue("district")
			report.Location = r.FormValue("location")
			report.PinCode = r.FormValue("pin_code")
			report.Description = r.FormValue("description")

			// Ignore any category sent by older clients.
			report.Category = ""

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

		// --------------------------------------------------------
		// Validate required fields
		// --------------------------------------------------------

		if strings.TrimSpace(report.Title) == "" ||
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

		pinCode := strings.TrimSpace(report.PinCode)

		if len(pinCode) != 6 {
			http.Error(
				w,
				"Pin code must contain exactly 6 digits",
				http.StatusBadRequest,
			)
			return
		}

		for _, char := range pinCode {

			if char < '0' || char > '9' {
				http.Error(
					w,
					"Pin code must contain exactly 6 digits",
					http.StatusBadRequest,
				)
				return
			}
		}

		// --------------------------------------------------------
		// Validate photos
		// --------------------------------------------------------

		const maxPhotoSize = 6 * 1024 * 1024

		for i, header := range photos {

			if header.Size > maxPhotoSize {
				http.Error(
					w,
					fmt.Sprintf("Photo %d exceeds 6 MB", i+1),
					http.StatusBadRequest,
				)
				return
			}

			ext := strings.ToLower(filepath.Ext(header.Filename))

			switch ext {

			case ".jpg", ".jpeg", ".png", ".webp":

			default:
				http.Error(
					w,
					fmt.Sprintf(
						"Photo %d has an unsupported file type",
						i+1,
					),
					http.StatusBadRequest,
				)
				return
			}
		}

		// --------------------------------------------------------
		// AI UNDERSTANDS + VALIDATES + CLASSIFIES + DEDUPLICATES
		// --------------------------------------------------------

		aiResult, err := analyzeReportWithAI(db, report)

		if err != nil {

			log.Println("AI analysis error:", err)

			http.Error(
				w,
				"AI service is currently unavailable. Please try again.",
				http.StatusServiceUnavailable,
			)
			return
		}

		// --------------------------------------------------------
		// AI DECISION
		// --------------------------------------------------------

		switch aiResult.Status {

		case "INVALID":

			w.Header().Set("Content-Type", "application/json")

			json.NewEncoder(w).Encode(map[string]any{
				"status":           "INVALID",
				"decision":         aiResult.Decision,
				"decision_reasons": aiResult.DecisionReasons,
				"analysis":         aiResult.Analysis,
			})

			return

		case "DUPLICATE_EXISTS":

			// IMPORTANT:
			// Do NOT insert the new report.
			//
			// Return the existing matching report information
			// so the frontend can point the citizen to it.

			w.Header().Set("Content-Type", "application/json")

			json.NewEncoder(w).Encode(map[string]any{
				"status":            "DUPLICATE_EXISTS",
				"decision":          aiResult.Decision,
				"decision_reasons":  aiResult.DecisionReasons,
				"analysis":          aiResult.Analysis,
				"duplicate_matches": aiResult.DuplicateMatches,
			})

			return

		case "NEED_HUMAN_REVIEW":

			w.Header().Set("Content-Type", "application/json")

			json.NewEncoder(w).Encode(map[string]any{
				"status":           "NEED_HUMAN_REVIEW",
				"decision":         aiResult.Decision,
				"decision_reasons": aiResult.DecisionReasons,
				"analysis":         aiResult.Analysis,
			})

			return

		case "VALID":

			// Continue to database persistence.

		default:

			log.Printf(
				"Unknown AI status: %q",
				aiResult.Status,
			)

			http.Error(
				w,
				"AI returned an invalid processing status",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// CATEGORY COMES EXCLUSIVELY FROM AI
		// --------------------------------------------------------

		report.Category = strings.TrimSpace(
			aiResult.Analysis.Category,
		)

		if report.Category == "" {

			log.Println(
				"AI returned VALID without a category",
			)

			http.Error(
				w,
				"AI could not determine a report category",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// GENERATE TRACK ID
		// --------------------------------------------------------

		trackID, err := generateTrackID()

		if err != nil {

			log.Println(
				"Track ID generation error:",
				err,
			)

			http.Error(
				w,
				"Failed to generate Track ID",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// INSERT VALID REPORT
		// --------------------------------------------------------

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

			log.Println(
				"Report creation error:",
				err,
			)

			http.Error(
				w,
				"Failed to create report",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// UPLOAD PHOTOS
		// --------------------------------------------------------

		photoURLs := []string{}

		for i, header := range photos {

			file, err := header.Open()

			if err != nil {

				log.Println(
					"Photo open error:",
					err,
				)

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

				log.Println(
					"Photo upload error:",
					err,
				)

				http.Error(
					w,
					"Failed to upload report photo",
					http.StatusInternalServerError,
				)
				return
			}

			photoURLs = append(
				photoURLs,
				url,
			)
		}

		// --------------------------------------------------------
		// SAVE PHOTO URLS
		// --------------------------------------------------------

		if len(photoURLs) > 0 {

			photoJSON, err := json.Marshal(
				photoURLs,
			)

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

				log.Println(
					"Photo URL database error:",
					err,
				)

				http.Error(
					w,
					"Failed to save photo information",
					http.StatusInternalServerError,
				)
				return
			}
		}

		// --------------------------------------------------------
		// SUCCESS RESPONSE
		// --------------------------------------------------------

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		json.NewEncoder(w).Encode(map[string]any{
			"id":               id,
			"track_id":         trackID,
			"status":           "VALID",
			"category":         report.Category,
			"ai_analysis":      aiResult.Analysis,
			"decision":         aiResult.Decision,
			"decision_reasons": aiResult.DecisionReasons,
			"photo_urls":       photoURLs,
		})
	}
}

// ============================================================
// VERIFY TRACK ID
// ============================================================

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

		if req.ReportID <= 0 ||
			strings.TrimSpace(req.TrackID) == "" {

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
			SELECT
				id,
				title,
				category,
				district,
				location,
				pin_code,
				description,
				status,
				created_at,
				photo_urls
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

			log.Println(
				"Track ID verification error:",
				err,
			)

			http.Error(
				w,
				"Failed to verify Track ID",
				http.StatusInternalServerError,
			)
			return
		}

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		json.NewEncoder(w).Encode(report)
	}
}



// ============================================================
// PUBLIC TRACK PROBLEM
// ============================================================

func getTrackProblem(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		trackID := strings.TrimSpace(r.PathValue("track_id"))

		if trackID == "" {
			http.Error(w, "Track ID is required", http.StatusBadRequest)
			return
		}

		// --------------------------------------------------------
		// Fetch problem using ONLY the Track ID
		// --------------------------------------------------------

		type Problem struct {
			ID             int64           `json:"id"`
			Title          string          `json:"title"`
			Category       string          `json:"category"`
			District       string          `json:"district"`
			Location       string          `json:"location"`
			PinCode        string          `json:"pin_code"`
			Description    string          `json:"description"`
			Status         string          `json:"status"`
			WorkflowStatus string          `json:"workflow_status"`
			CreatedAt      time.Time       `json:"created_at"`
			PhotoURLs      json.RawMessage `json:"photo_urls"`
		}

		var problem Problem

		err := db.QueryRow(
			r.Context(),
			`
			SELECT
				id,
				title,
				category,
				district,
				location,
				pin_code,
				description,
				status,
				workflow_status,
				created_at,
				photo_urls
			FROM public.reports
			WHERE track_id = $1
			`,
			trackID,
		).Scan(
			&problem.ID,
			&problem.Title,
			&problem.Category,
			&problem.District,
			&problem.Location,
			&problem.PinCode,
			&problem.Description,
			&problem.Status,
			&problem.WorkflowStatus,
			&problem.CreatedAt,
			&problem.PhotoURLs,
		)

		if err != nil {

			if err == pgx.ErrNoRows {
				http.Error(
					w,
					"Invalid Track ID",
					http.StatusNotFound,
				)
				return
			}

			log.Println("Track problem lookup error:", err)

			http.Error(
				w,
				"Failed to fetch tracked problem",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// Public team member structure
		// --------------------------------------------------------

		type Member struct {
			Name  string `json:"name"`
			Role  string `json:"role"`
			Email string `json:"email"`
		}

		// --------------------------------------------------------
		// Default: no team yet
		// --------------------------------------------------------

		students := make([]Member, 0)
		mentors := make([]Member, 0)

		var workID int64
		var workStatus string

		err = db.QueryRow(
			r.Context(),
			`
			SELECT
				id,
				status
			FROM works
			WHERE report_id = $1
			`,
			problem.ID,
		).Scan(
			&workID,
			&workStatus,
		)

		// No work means the problem has not been taken yet.
		if err != nil && err != pgx.ErrNoRows {
			log.Println("Track work lookup error:", err)

			http.Error(
				w,
				"Failed to fetch team status",
				http.StatusInternalServerError,
			)
			return
		}

		workExists := err == nil

		// --------------------------------------------------------
		// Fetch students if a Work exists
		// --------------------------------------------------------

		if workExists {

			studentRows, err := db.Query(
				r.Context(),
				`
				SELECT
					sa.full_name,
					sa.role,
					sa.email
				FROM work_students ws
				JOIN solver_accounts sa
					ON sa.id = ws.student_account_id
				WHERE ws.work_id = $1
				ORDER BY ws.created_at ASC
				`,
				workID,
			)

			if err != nil {
				http.Error(
					w,
					"Failed to fetch team students",
					http.StatusInternalServerError,
				)
				return
			}

			defer studentRows.Close()

			for studentRows.Next() {

				var member Member

				if err := studentRows.Scan(
					&member.Name,
					&member.Role,
					&member.Email,
				); err != nil {
					http.Error(
						w,
						"Failed to read team students",
						http.StatusInternalServerError,
					)
					return
				}

				students = append(students, member)
			}

			if err := studentRows.Err(); err != nil {
				http.Error(
					w,
					"Failed to read team students",
					http.StatusInternalServerError,
				)
				return
			}
		}

		// --------------------------------------------------------
		// Fetch mentors assigned to this problem
		// --------------------------------------------------------

		mentorRows, err := db.Query(
			r.Context(),
			`
			SELECT
				sa.full_name,
				sa.role,
				sa.email
			FROM problem_mentors pm
			JOIN solver_accounts sa
				ON sa.id = pm.account_id
			WHERE pm.report_id = $1
			ORDER BY pm.created_at ASC
			`,
			problem.ID,
		)

		if err != nil {
			http.Error(
				w,
				"Failed to fetch problem mentors",
				http.StatusInternalServerError,
			)
			return
		}

		defer mentorRows.Close()

		for mentorRows.Next() {

			var member Member

			if err := mentorRows.Scan(
				&member.Name,
				&member.Role,
				&member.Email,
			); err != nil {
				http.Error(
					w,
					"Failed to read problem mentors",
					http.StatusInternalServerError,
				)
				return
			}

			mentors = append(mentors, member)
		}

		if err := mentorRows.Err(); err != nil {
			http.Error(
				w,
				"Failed to read problem mentors",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// Determine public status
		// --------------------------------------------------------

		publicStatus := "AVAILABLE"

		if strings.EqualFold(problem.WorkflowStatus, "SOLVED") ||
			strings.EqualFold(workStatus, "SOLVED") ||
			strings.EqualFold(workStatus, "RESOLVED") {

			publicStatus = "SOLVED"

		} else if workExists {

			publicStatus = "IN_PROGRESS"
		}

		// --------------------------------------------------------
		// Response
		// --------------------------------------------------------

		response := map[string]any{
			"track_id": trackID,

			"problem": problem,

			"status": publicStatus,

			"team": map[string]any{
				"exists":        workExists,
				"status":        workStatus,
				"student_count": len(students),
				"max_students":  4,
				"mentor_count":  len(mentors),
				"max_mentors":   2,
				"students":      students,
				"mentors":       mentors,
			},
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(response)
	}
}

// ============================================================
// GET ALL REPORTS
// ============================================================

func getReports(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		if r.Method != http.MethodGet {
			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		rows, err := db.Query(
			context.Background(),
			`
			SELECT
				id,
				title,
				category,
				district,
				location,
				pin_code,
				description,
				status,
				created_at,
				photo_urls
			FROM public.reports
			ORDER BY created_at DESC
			`,
		)

		if err != nil {

			log.Println(
				"Report query error:",
				err,
			)

			http.Error(
				w,
				"Failed to fetch reports",
				http.StatusInternalServerError,
			)
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

				log.Println(
					"Report scan error:",
					err,
				)

				http.Error(
					w,
					"Failed to read reports",
					http.StatusInternalServerError,
				)
				return
			}

			reports = append(
				reports,
				report,
			)
		}

		if err := rows.Err(); err != nil {

			http.Error(
				w,
				"Failed to read reports",
				http.StatusInternalServerError,
			)
			return
		}

		json.NewEncoder(w).Encode(reports)
	}
}

// ============================================================
// GET REPORT BY ID
// ============================================================

func getReportByID(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		if r.Method != http.MethodGet {
			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

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
			SELECT
				id,
				title,
				category,
				district,
				location,
				pin_code,
				description,
				status,
				created_at
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

			http.Error(
				w,
				"Report not found",
				http.StatusNotFound,
			)
			return
		}

		json.NewEncoder(w).Encode(report)
	}
}

// ============================================================
// LOGIN
// ============================================================

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func login(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Access-Control-Allow-Methods",
			"POST, OPTIONS",
		)

		w.Header().Set(
			"Access-Control-Allow-Headers",
			"Content-Type",
		)

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		var req LoginRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(
				w,
				"Invalid JSON",
				http.StatusBadRequest,
			)
			return
		}

		if req.Email == "" ||
			req.Password == "" {

			http.Error(
				w,
				"Email and password are required",
				http.StatusBadRequest,
			)
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
			SELECT
				id,
				full_name,
				role,
				password_hash
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

			http.Error(
				w,
				"Invalid email or password",
				http.StatusUnauthorized,
			)
			return
		}

		if err := bcrypt.CompareHashAndPassword(
			[]byte(passwordHash),
			[]byte(req.Password),
		); err != nil {

			http.Error(
				w,
				"Invalid email or password",
				http.StatusUnauthorized,
			)
			return
		}

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		json.NewEncoder(w).Encode(map[string]any{
			"id":        id,
			"full_name": fullName,
			"role":      role,
			"message":   "Login successful",
		})
	}
}

// ============================================================
// SOLVER PROFILE
// ============================================================

func getSolverProfile(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Content-Type",
			"application/json",
		)

		if r.Method != http.MethodGet {
			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		accountIDStr := strings.TrimPrefix(
			r.URL.Path,
			"/api/solver/profile/",
		)

		accountID, err := strconv.ParseInt(
			accountIDStr,
			10,
			64,
		)

		if err != nil {
			http.Error(
				w,
				"Invalid account ID",
				http.StatusBadRequest,
			)
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
				http.Error(
					w,
					"Profile not found",
					http.StatusNotFound,
				)
				return
			}

			log.Println(
				"Profile lookup error:",
				err,
			)

			http.Error(
				w,
				"Failed to fetch profile",
				http.StatusInternalServerError,
			)
			return
		}

		json.NewEncoder(w).Encode(profile)
	}
}

func saveSolverProfile(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set(
			"Access-Control-Allow-Origin",
			"*",
		)

		w.Header().Set(
			"Access-Control-Allow-Methods",
			"POST, PUT, OPTIONS",
		)

		w.Header().Set(
			"Access-Control-Allow-Headers",
			"Content-Type",
		)

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if r.Method != http.MethodPost &&
			r.Method != http.MethodPut {

			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
			return
		}

		accountIDStr := strings.TrimPrefix(
			r.URL.Path,
			"/api/solver/profile/",
		)

		accountID, err := strconv.ParseInt(
			accountIDStr,
			10,
			64,
		)

		if err != nil {
			http.Error(
				w,
				"Invalid account ID",
				http.StatusBadRequest,
			)
			return
		}

		var req SolverProfileRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(
				w,
				"Invalid JSON",
				http.StatusBadRequest,
			)
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

			log.Println(
				"Profile save error:",
				err,
			)

			http.Error(
				w,
				"Failed to save profile",
				http.StatusInternalServerError,
			)
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

			http.Error(
				w,
				"Method not allowed",
				http.StatusMethodNotAllowed,
			)
		}
	}
}

// ============================================================
// SUBMIT PROPOSAL
// ============================================================

type ProposalRequest struct {
	StudentAccountID int64  `json:"student_account_id"`
	ProposalText     string `json:"proposal_text"`
}

func createProposal(db *pgxpool.Pool) http.HandlerFunc {
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

		// --------------------------------------------------------
		// Get report ID from URL
		// --------------------------------------------------------

		reportIDStr := r.PathValue("id")

		reportID, err := strconv.ParseInt(reportIDStr, 10, 64)
		if err != nil {
			http.Error(w, "Invalid report ID", http.StatusBadRequest)
			return
		}

		// --------------------------------------------------------
		// Parse request
		// --------------------------------------------------------

		var req ProposalRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		req.ProposalText = strings.TrimSpace(req.ProposalText)

		if req.StudentAccountID <= 0 {
			http.Error(w, "Invalid student account ID", http.StatusBadRequest)
			return
		}

		if req.ProposalText == "" {
			http.Error(w, "Proposal text is required", http.StatusBadRequest)
			return
		}

		// --------------------------------------------------------
		// Verify account role
		// --------------------------------------------------------

		var role string

		err = db.QueryRow(
			context.Background(),
			`
			SELECT role
			FROM solver_accounts
			WHERE id = $1
			`,
			req.StudentAccountID,
		).Scan(&role)

		if err != nil {
			if err == pgx.ErrNoRows {
				http.Error(w, "Account not found", http.StatusNotFound)
				return
			}

			log.Println("Account lookup error:", err)
			http.Error(w, "Failed to verify account", http.StatusInternalServerError)
			return
		}

		if role != "student" && role != "researcher" {
			http.Error(
				w,
				"Only students and researchers can submit proposals",
				http.StatusForbidden,
			)
			return
		}

		// --------------------------------------------------------
		// Verify report and workflow state
		// --------------------------------------------------------

		var workflowStatus string

		err = db.QueryRow(
			context.Background(),
			`
			SELECT workflow_status
			FROM reports
			WHERE id = $1
			`,
			reportID,
		).Scan(&workflowStatus)

		if err != nil {
			if err == pgx.ErrNoRows {
				http.Error(w, "Problem not found", http.StatusNotFound)
				return
			}

			log.Println("Report lookup error:", err)
			http.Error(w, "Failed to verify problem", http.StatusInternalServerError)
			return
		}

		if workflowStatus == "SOLVED" {
			http.Error(
				w,
				"This problem is already solved",
				http.StatusConflict,
			)
			return
		}

		// --------------------------------------------------------
		// Check accepted student capacity
		// --------------------------------------------------------

		var acceptedCount int

		err = db.QueryRow(
			context.Background(),
			`
			SELECT COUNT(*)
			FROM proposals
			WHERE report_id = $1
			  AND status = 'ACCEPTED'
			`,
			reportID,
		).Scan(&acceptedCount)

		if err != nil {
			log.Println("Accepted proposal count error:", err)
			http.Error(
				w,
				"Failed to check team capacity",
				http.StatusInternalServerError,
			)
			return
		}

		if acceptedCount >= 4 {
			http.Error(
				w,
				"This problem already has 4 students",
				http.StatusConflict,
			)
			return
		}

		// --------------------------------------------------------
		// Insert proposal
		// --------------------------------------------------------

		var proposalID int64

		err = db.QueryRow(
			context.Background(),
			`
			INSERT INTO proposals (
				report_id,
				student_account_id,
				proposal_text
			)
			VALUES ($1, $2, $3)
			RETURNING id
			`,
			reportID,
			req.StudentAccountID,
			req.ProposalText,
		).Scan(&proposalID)

		if err != nil {

			// Same student already proposed for this problem
			if strings.Contains(err.Error(), "proposals_unique_student_problem") {
				http.Error(
					w,
					"You have already submitted a proposal for this problem",
					http.StatusConflict,
				)
				return
			}

			log.Println("Proposal creation error:", err)
			http.Error(
				w,
				"Failed to submit proposal",
				http.StatusInternalServerError,
			)
			return
		}

		// --------------------------------------------------------
		// Response
		// --------------------------------------------------------

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(map[string]any{
			"id":                 proposalID,
			"report_id":          reportID,
			"student_account_id": req.StudentAccountID,
			"status":             "PENDING",
			"message":            "Proposal submitted successfully",
		})
	}
}

func getProposals(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		reportID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil || reportID <= 0 {
			http.Error(w, "Invalid report ID", http.StatusBadRequest)
			return
		}

		rows, err := db.Query(r.Context(), `
			SELECT
				id,
				report_id,
				student_account_id,
				proposal_text,
				status,
				created_at
			FROM proposals
			WHERE report_id = $1
			ORDER BY created_at ASC
		`, reportID)

		if err != nil {
			http.Error(w, "Failed to fetch proposals", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		type Proposal struct {
			ID               int64     `json:"id"`
			ReportID         int64     `json:"report_id"`
			StudentAccountID int64     `json:"student_account_id"`
			ProposalText     string    `json:"proposal_text"`
			Status           string    `json:"status"`
			CreatedAt        time.Time `json:"created_at"`
		}

		proposals := make([]Proposal, 0)

		for rows.Next() {
			var p Proposal

			err := rows.Scan(
				&p.ID,
				&p.ReportID,
				&p.StudentAccountID,
				&p.ProposalText,
				&p.Status,
				&p.CreatedAt,
			)

			if err != nil {
				http.Error(w, "Failed to read proposals", http.StatusInternalServerError)
				return
			}

			proposals = append(proposals, p)
		}

		if err := rows.Err(); err != nil {
			http.Error(w, "Failed to read proposals", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(proposals)
	}
}

func joinProblemAsMentor(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		reportID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil || reportID <= 0 {
			http.Error(w, "Invalid report ID", http.StatusBadRequest)
			return
		}

		type MentorRequest struct {
			AccountID int64 `json:"account_id"`
		}

		var req MentorRequest

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.AccountID <= 0 {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		var (
			success  bool
			message  string
			mentorID int64
		)

		err = db.QueryRow(r.Context(), `
			SELECT success, message, mentor_id
			FROM join_problem_as_mentor($1, $2)
		`, reportID, req.AccountID).Scan(
			&success,
			&message,
			&mentorID,
		)

		if err != nil {
			http.Error(w, "Failed to join problem as mentor", http.StatusInternalServerError)
			return
		}

		status := http.StatusOK
		if !success {
			status = http.StatusConflict
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":    success,
			"message":    message,
			"mentor_id":  mentorID,
			"report_id":  reportID,
			"account_id": req.AccountID,
		})
	}
}

func acceptProposal(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		proposalID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil || proposalID <= 0 {
			http.Error(w, "Invalid proposal ID", http.StatusBadRequest)
			return
		}

		var req struct {
			AccountID int64 `json:"account_id"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.AccountID <= 0 {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		var (
			success bool
			message string
			workID  *int64
		)

		err = db.QueryRow(r.Context(), `
			SELECT success, message, work_id
			FROM accept_proposal($1, $2)
		`, proposalID, req.AccountID).Scan(
			&success,
			&message,
			&workID,
		)

		if err != nil {
			http.Error(w, "Failed to accept proposal", http.StatusInternalServerError)
			return
		}

		status := http.StatusOK
		if !success {
			status = http.StatusConflict
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":     success,
			"message":     message,
			"proposal_id": proposalID,
			"account_id":  req.AccountID,
			"work_id":     workID,
		})
	}
}

func rejectProposal(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		proposalID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil || proposalID <= 0 {
			http.Error(w, "Invalid proposal ID", http.StatusBadRequest)
			return
		}

		var req struct {
			AccountID int64 `json:"account_id"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.AccountID <= 0 {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		var (
			success bool
			message string
		)

		err = db.QueryRow(r.Context(), `
			SELECT success, message
			FROM reject_proposal($1, $2)
		`, proposalID, req.AccountID).Scan(
			&success,
			&message,
		)

		if err != nil {
			http.Error(w, "Failed to reject proposal", http.StatusInternalServerError)
			return
		}

		status := http.StatusOK
		if !success {
			status = http.StatusConflict
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)

		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":     success,
			"message":     message,
			"proposal_id": proposalID,
			"account_id":  req.AccountID,
		})
	}
}

func getWork(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		reportID, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil || reportID <= 0 {
			http.Error(w, "Invalid report ID", http.StatusBadRequest)
			return
		}

		type Student struct {
			AccountID    int64  `json:"account_id"`
			FullName     string `json:"full_name"`
			University   string `json:"university"`
			EnrollmentID string `json:"enrollment_id"`
			Department   string `json:"department"`
			YearOfStudy  string `json:"year_of_study"`
			Skills       string `json:"skills"`
			Interests    string `json:"interests"`
			Projects     string `json:"projects"`
			ProposalText string `json:"proposal_text"`
		}

		type Mentor struct {
			AccountID   int64  `json:"account_id"`
			FullName    string `json:"full_name"`
			Institution string `json:"institution"`
			Phone       string `json:"phone"`
			University  string `json:"university"`
			Department  string `json:"department"`
			Skills      string `json:"skills"`
			Interests   string `json:"interests"`
			Projects    string `json:"projects"`
		}

		type Problem struct {
			ID             int64       `json:"id"`
			Title          string      `json:"title"`
			Category       string      `json:"category"`
			District       string      `json:"district"`
			Location       string      `json:"location"`
			PinCode        string      `json:"pin_code"`
			Description    string      `json:"description"`
			Status         string      `json:"status"`
			WorkflowStatus string      `json:"workflow_status"`
			CreatedAt      time.Time   `json:"created_at"`
			TrackID        *string     `json:"track_id"`
			PhotoURLs      interface{} `json:"photo_urls"`
		}

		type Work struct {
			WorkID       int64     `json:"work_id"`
			ReportID     int64     `json:"report_id"`
			Status       string    `json:"status"`
			CreatedAt    time.Time `json:"created_at"`
			StudentCount int       `json:"student_count"`
			MaxStudents  int       `json:"max_students"`
			MentorCount  int       `json:"mentor_count"`
			MaxMentors   int       `json:"max_mentors"`
			Problem      Problem   `json:"problem"`
			Students     []Student `json:"students"`
			Mentors      []Mentor  `json:"mentors"`
		}

		var work Work

		err = db.QueryRow(r.Context(), `
			SELECT
				w.id,
				w.report_id,
				w.status,
				w.created_at,

				r.id,
				r.title,
				r.category,
				r.district,
				r.location,
				r.pin_code,
				r.description,
				r.status,
				r.workflow_status,
				r.created_at,
				r.track_id,
				r.photo_urls

			FROM works w
			JOIN reports r ON r.id = w.report_id
			WHERE w.report_id = $1
		`, reportID).Scan(
			&work.WorkID,
			&work.ReportID,
			&work.Status,
			&work.CreatedAt,

			&work.Problem.ID,
			&work.Problem.Title,
			&work.Problem.Category,
			&work.Problem.District,
			&work.Problem.Location,
			&work.Problem.PinCode,
			&work.Problem.Description,
			&work.Problem.Status,
			&work.Problem.WorkflowStatus,
			&work.Problem.CreatedAt,
			&work.Problem.TrackID,
			&work.Problem.PhotoURLs,
		)

		if err != nil {
			if err == pgx.ErrNoRows {
				http.Error(w, "Work not found for this problem", http.StatusNotFound)
				return
			}

			http.Error(w, "Failed to fetch work", http.StatusInternalServerError)
			return
		}

		// Get students belonging to this Work.
		studentRows, err := db.Query(r.Context(), `
			SELECT
				sa.id,
				COALESCE(sp.full_name, sa.full_name, ''),
				COALESCE(sp.university, ''),
				COALESCE(sp.enrollment_id, ''),
				COALESCE(sp.department, ''),
				COALESCE(sp.year_of_study::text, ''),
				COALESCE(sp.skills, ''),
				COALESCE(sp.interests, ''),
				COALESCE(sp.projects, ''),
				COALESCE(p.proposal_text, '')

			FROM work_students ws
			JOIN solver_accounts sa
				ON sa.id = ws.student_account_id
			LEFT JOIN solver_profiles sp
				ON sp.account_id = sa.id
			LEFT JOIN proposals p
				ON p.report_id = $1
				AND p.student_account_id = sa.id
				AND p.status = 'ACCEPTED'

			WHERE ws.work_id = $2
			ORDER BY ws.created_at ASC
		`, reportID, work.WorkID)

		if err != nil {
			http.Error(w, "Failed to fetch work students", http.StatusInternalServerError)
			return
		}
		defer studentRows.Close()

		work.Students = make([]Student, 0)

		for studentRows.Next() {
			var student Student

			if err := studentRows.Scan(
				&student.AccountID,
				&student.FullName,
				&student.University,
				&student.EnrollmentID,
				&student.Department,
				&student.YearOfStudy,
				&student.Skills,
				&student.Interests,
				&student.Projects,
				&student.ProposalText,
			); err != nil {
				http.Error(w, "Failed to read work students", http.StatusInternalServerError)
				return
			}

			work.Students = append(work.Students, student)
		}

		if err := studentRows.Err(); err != nil {
			http.Error(w, "Failed to read work students", http.StatusInternalServerError)
			return
		}

		// Get mentors assigned to this problem.
		mentorRows, err := db.Query(r.Context(), `
			SELECT
				sa.id,
				COALESCE(sp.full_name, sa.full_name, ''),
				COALESCE(sa.institution, ''),
				COALESCE(sp.phone, sa.phone, ''),
				COALESCE(sp.university, ''),
				COALESCE(sp.department, ''),
				COALESCE(sp.skills, ''),
				COALESCE(sp.interests, ''),
				COALESCE(sp.projects, '')

			FROM problem_mentors pm
			JOIN solver_accounts sa
				ON sa.id = pm.account_id
			LEFT JOIN solver_profiles sp
				ON sp.account_id = sa.id

			WHERE pm.report_id = $1
			ORDER BY pm.created_at ASC
		`, reportID)

		if err != nil {
			http.Error(w, "Failed to fetch work mentors", http.StatusInternalServerError)
			return
		}
		defer mentorRows.Close()

		work.Mentors = make([]Mentor, 0)

		for mentorRows.Next() {
			var mentor Mentor

			if err := mentorRows.Scan(
				&mentor.AccountID,
				&mentor.FullName,
				&mentor.Institution,
				&mentor.Phone,
				&mentor.University,
				&mentor.Department,
				&mentor.Skills,
				&mentor.Interests,
				&mentor.Projects,
			); err != nil {
				http.Error(w, "Failed to read work mentors", http.StatusInternalServerError)
				return
			}

			work.Mentors = append(work.Mentors, mentor)
		}

		if err := mentorRows.Err(); err != nil {
			http.Error(w, "Failed to read work mentors", http.StatusInternalServerError)
			return
		}

		work.StudentCount = len(work.Students)
		work.MaxStudents = 4
		work.MentorCount = len(work.Mentors)
		work.MaxMentors = 2

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(work)
	}
}

func getProjects(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		accountIDStr := r.URL.Query().Get("account_id")
		if accountIDStr == "" {
			http.Error(w, "account_id is required", http.StatusBadRequest)
			return
		}

		accountID, err := strconv.ParseInt(accountIDStr, 10, 64)
		if err != nil || accountID <= 0 {
			http.Error(w, "Invalid account ID", http.StatusBadRequest)
			return
		}

		rows, err := db.Query(
			context.Background(),
			`
			SELECT DISTINCT
				w.id,
				w.report_id,
				w.status,
				w.created_at,
				r.title,
				r.category,
				r.district,
				r.location,
				r.description,
				r.workflow_status
			FROM works w
			JOIN reports r
				ON r.id = w.report_id
			WHERE EXISTS (
				SELECT 1
				FROM work_students ws
				WHERE ws.work_id = w.id
				  AND ws.student_account_id = $1
			)
			OR EXISTS (
				SELECT 1
				FROM problem_mentors pm
				WHERE pm.report_id = w.report_id
				  AND pm.account_id = $1
			)
			ORDER BY w.created_at DESC
			`,
			accountID,
		)
		if err != nil {
			http.Error(w, "Failed to fetch projects", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		type Member struct {
			AccountID int64  `json:"account_id"`
			FullName  string `json:"full_name"`
			Email     string `json:"email"`
			Role      string `json:"role"`
		}

		type Project struct {
			WorkID         int64     `json:"work_id"`
			ReportID       int64     `json:"report_id"`
			Status         string    `json:"status"`
			CreatedAt      time.Time `json:"created_at"`
			Title          string    `json:"title"`
			Category       string    `json:"category"`
			District       string    `json:"district"`
			Location       string    `json:"location"`
			Description    string    `json:"description"`
			WorkflowStatus string    `json:"workflow_status"`

			StudentCount int `json:"student_count"`
			MentorCount  int `json:"mentor_count"`
			MaxStudents  int `json:"max_students"`
			MaxMentors   int `json:"max_mentors"`

			Members []Member `json:"members"`
		}

		projects := make([]Project, 0)

		for rows.Next() {
			var p Project

			if err := rows.Scan(
				&p.WorkID,
				&p.ReportID,
				&p.Status,
				&p.CreatedAt,
				&p.Title,
				&p.Category,
				&p.District,
				&p.Location,
				&p.Description,
				&p.WorkflowStatus,
			); err != nil {
				http.Error(w, "Failed to read projects", http.StatusInternalServerError)
				return
			}

			// ---------------------------------------------------------
			// Get students working on this Work
			// ---------------------------------------------------------

			memberRows, err := db.Query(
				context.Background(),
				`
				SELECT
					sa.id,
					sa.full_name,
					sa.email,
					sa.role
				FROM work_students ws
				JOIN solver_accounts sa
					ON sa.id = ws.student_account_id
				WHERE ws.work_id = $1
				ORDER BY ws.created_at
				`,
				p.WorkID,
			)
			if err != nil {
				http.Error(
					w,
					"Failed to fetch project students",
					http.StatusInternalServerError,
				)
				return
			}

			for memberRows.Next() {
				var member Member

				if err := memberRows.Scan(
					&member.AccountID,
					&member.FullName,
					&member.Email,
					&member.Role,
				); err != nil {
					memberRows.Close()
					http.Error(
						w,
						"Failed to read project students",
						http.StatusInternalServerError,
					)
					return
				}

				p.Members = append(p.Members, member)
			}

			if err := memberRows.Err(); err != nil {
				memberRows.Close()
				http.Error(
					w,
					"Failed to read project students",
					http.StatusInternalServerError,
				)
				return
			}

			memberRows.Close()

			// ---------------------------------------------------------
			// Get company mentors assigned to this problem
			// ---------------------------------------------------------

			mentorRows, err := db.Query(
				context.Background(),
				`
				SELECT
					sa.id,
					sa.full_name,
					sa.email,
					sa.role
				FROM problem_mentors pm
				JOIN solver_accounts sa
					ON sa.id = pm.account_id
				WHERE pm.report_id = $1
				ORDER BY pm.created_at
				`,
				p.ReportID,
			)
			if err != nil {
				http.Error(
					w,
					"Failed to fetch project mentors",
					http.StatusInternalServerError,
				)
				return
			}

			for mentorRows.Next() {
				var member Member

				if err := mentorRows.Scan(
					&member.AccountID,
					&member.FullName,
					&member.Email,
					&member.Role,
				); err != nil {
					mentorRows.Close()
					http.Error(
						w,
						"Failed to read project mentors",
						http.StatusInternalServerError,
					)
					return
				}

				p.Members = append(p.Members, member)
			}

			if err := mentorRows.Err(); err != nil {
				mentorRows.Close()
				http.Error(
					w,
					"Failed to read project mentors",
					http.StatusInternalServerError,
				)
				return
			}

			mentorRows.Close()

			// ---------------------------------------------------------
			// Count students and mentors separately.
			//
			// IMPORTANT:
			// The maximum of 4 applies to students.
			// The maximum of 2 applies to company mentors.
			// Mentors therefore do NOT consume student slots.
			// ---------------------------------------------------------

			p.StudentCount = 0
			p.MentorCount = 0

			for _, member := range p.Members {
				if member.Role == "company" {
					p.MentorCount++
				} else if member.Role == "student" || member.Role == "researcher" {
					p.StudentCount++
				}
			}

			p.MaxStudents = 4
			p.MaxMentors = 2

			projects = append(projects, p)
		}

		if err := rows.Err(); err != nil {
			http.Error(
				w,
				"Failed to read projects",
				http.StatusInternalServerError,
			)
			return
		}

		json.NewEncoder(w).Encode(map[string]any{
			"account_id": accountID,
			"projects":   projects,
			"count":      len(projects),
		})
	}
}

func getProblemMentors(db *pgxpool.Pool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		reportID, err := strconv.ParseInt(
			r.PathValue("id"),
			10,
			64,
		)

		if err != nil || reportID <= 0 {
			http.Error(w, "Invalid report ID", http.StatusBadRequest)
			return
		}

		type Mentor struct {
			AccountID int64  `json:"account_id"`
			FullName  string `json:"full_name"`
			Email     string `json:"email"`
			Role      string `json:"role"`
		}

		mentors := []Mentor{}

		rows, err := db.Query(
			context.Background(),
			`
			SELECT
				pm.account_id,
				sa.full_name,
				sa.email,
				sa.role
			FROM problem_mentors pm
			JOIN solver_accounts sa
				ON sa.id = pm.account_id
			WHERE pm.report_id = $1
			ORDER BY pm.created_at ASC
			`,
			reportID,
		)

		if err != nil {
			log.Println("Problem mentor lookup error:", err)
			http.Error(
				w,
				"Failed to fetch problem mentors",
				http.StatusInternalServerError,
			)
			return
		}

		defer rows.Close()

		for rows.Next() {
			var mentor Mentor

			if err := rows.Scan(
				&mentor.AccountID,
				&mentor.FullName,
				&mentor.Email,
				&mentor.Role,
			); err != nil {
				log.Println("Problem mentor scan error:", err)
				http.Error(
					w,
					"Failed to read problem mentors",
					http.StatusInternalServerError,
				)
				return
			}

			mentors = append(mentors, mentor)
		}

		if err := rows.Err(); err != nil {
			log.Println("Problem mentor rows error:", err)
			http.Error(
				w,
				"Failed to read problem mentors",
				http.StatusInternalServerError,
			)
			return
		}

		json.NewEncoder(w).Encode(map[string]any{
			"report_id":    reportID,
			"mentor_count": len(mentors),
			"mentors":      mentors,
		})
	}
}
