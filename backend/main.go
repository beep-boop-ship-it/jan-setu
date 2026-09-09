package main

import (
	"context"
	"log"
	"net/http"
	"os"
)

func main() {
	db, err := connectDB()
	if err != nil {
		log.Fatal("Database connection failed:", err)
	}
	defer db.Close()

	err = db.Ping(context.Background())
	if err != nil {
		log.Fatal("Database ping failed:", err)
	}

	log.Println("PostgreSQL connected successfully!")

http.HandleFunc("/api/auth/signup", signup(db))
http.HandleFunc("/api/auth/login", login(db))
http.HandleFunc("/api/solver/profile/", solverProfile(db))
http.HandleFunc("/api/reports", createReport(db))
http.HandleFunc("/api/reports/verify-track", verifyTrackID(db))
http.HandleFunc("/api/reports/all", getReports(db))
http.HandleFunc("/api/reports/{id}", getReportByID(db))
http.HandleFunc("/api/proposals/{id}/accept", acceptProposal(db))
http.HandleFunc("/api/proposals/{id}/reject", rejectProposal(db))
http.HandleFunc("/api/projects", getProjects(db))
http.HandleFunc("GET /api/reports/{id}/mentors", getProblemMentors(db))
http.HandleFunc(
	"GET /api/industry/matches/{accountID}",
	getIndustryMatches(db),
)
http.HandleFunc("/api/reports/{id}/work", getWork(db))
http.HandleFunc("/api/reports/{id}/mentors", joinProblemAsMentor(db))
http.HandleFunc("/api/reports/{id}/proposals", func(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost || r.Method == http.MethodOptions {
		createProposal(db)(w, r)
		return
	}

	if r.Method == http.MethodGet {
		getProposals(db)(w, r)
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
})

	http.Handle("/", http.FileServer(http.Dir("../frontend")))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Println("Server running on http://localhost:" + port)

	log.Fatal(http.ListenAndServe(":"+port, nil))
}
