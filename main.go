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

	http.HandleFunc("/api/reports", createReport(db))
	http.Handle("/", http.FileServer(http.Dir(".")))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Println("Server running on http://localhost:" + port)

	log.Fatal(http.ListenAndServe(":"+port, nil))
}