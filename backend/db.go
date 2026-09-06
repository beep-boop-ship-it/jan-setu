package main

import (
	"context"
	"os"

	"github.com/jackc/pgx/v5/pgxpool"
)

func connectDB() (*pgxpool.Pool, error) {
	databaseURL := os.Getenv("DATABASE_URL")

	if databaseURL == "" {
		return nil, os.ErrNotExist
	}

	return pgxpool.New(context.Background(), databaseURL)
}
