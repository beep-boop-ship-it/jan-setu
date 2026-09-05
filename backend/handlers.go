package main

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/jackc/pgx/v5/pgxpool"
)

type Report struct {
	Title       string `json:"title"`
	Category    string `json:"category"`
	District    string `json:"district"`
	Location    string `json:"location"`
	PinCode     string `json:"pin_code"`
	Description string `json:"description"`
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

		err := json.NewDecoder(r.Body).Decode(&report)
		if err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		var id int64

		err = db.QueryRow(
			context.Background(),
			`
			INSERT INTO reports
				(title, category, district, location, pin_code, description)
			VALUES
				($1, $2, $3, $4, $5, $6)
			RETURNING id
			`,
			report.Title,
			report.Category,
			report.District,
			report.Location,
			report.PinCode,
			report.Description,
		).Scan(&id)

		if err != nil {
			http.Error(w, "Failed to create report", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")

		json.NewEncoder(w).Encode(map[string]any{
			"id":     id,
			"status": "submitted",
		})
	}
}