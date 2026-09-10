# JAN SETU

**Live Prototype:** https://jan-setu-bf03.onrender.com/

## Problem Statement — PS43

PS43 focuses on connecting **citizens, government, universities and industry** to identify societal problems and turn them into practical solutions.

The challenge is not just reporting a problem, but creating a clear path from the **problem to the people who can solve it**.

## Our Solution

JAN SETU provides this workflow in one platform.

**Report → Match → Solve → Track**

Citizens can report problems with details, location and supporting images. The system processes the problem, identifies relevant expertise and connects it with potential solvers and industry experts.

## Demo Video

[▶ Watch JAN SETU Demo]

https://github.com/user-attachments/assets/ddb2223c-2945-4817-8e47-c3392d5ab797





## Problem Flow

<img width="1280" height="388" alt="flowchart_sih" src="https://github.com/user-attachments/assets/e45c71e8-a939-4cf7-bc44-92675c667d49" />

## Key Features

* Citizen problem reporting
* Location and image-based reporting
* Problem browsing and tracking
* Solver profiles
* AI-based problem-to-solver matching
* Industry expert workflow
* Solution proposal and review
* Report validation and error handling
* Duplicate problem detection

## AI Matching

<img width="1600" height="953" alt="ai-matching" src="https://github.com/user-attachments/assets/7c0c8cbd-ec6c-45d6-8237-cf785240b935" />

The AI matching system connects reported problems with relevant solvers based on the problem requirements and solver expertise.

## Report Handling & Error States

<img width="945" height="464" alt="gibberish_handle" src="https://github.com/user-attachments/assets/9024bc46-20ad-45a3-acd5-a035ed20a104" />

JAN SETU handles invalid or incomplete reports before they enter the main workflow.

Examples include:

* Missing required information
* Invalid report submissions
* Invalid or unsupported images
* Gibberish or meaningless reports
* Backend or database errors

## Duplicate Handling

<img width="1146" height="472" alt="duplicate_sih" src="https://github.com/user-attachments/assets/90bf7225-202c-454e-9d65-f11fd402d30e" />

JAN SETU also checks for duplicate problems so that the same issue does not unnecessarily enter the workflow multiple times.

## Technology Stack

| Part       | Technology            |
| ---------- | --------------------- |
| Frontend   | HTML, CSS, JavaScript |
| Backend    | Go, REST APIs         |
| AI         | Python                |
| Database   | PostgreSQL / Supabase |
| Deployment | Render                |

## Project Structure

```text
jan-setu/
├── frontend/
├── backend/
├── ai/
├── .github/
└── pyproject.toml
```

## Running Locally

### Backend

```bash
cd backend
go run .
```

### AI

Install the Python dependencies defined in `pyproject.toml` and run the required AI services.

### Frontend

Run the frontend using a local web server.

## Prototype

This repository contains our working prototype developed for **Smart India Hackathon 2026 — PS43**.

**Live Prototype:** https://jan-setu-bf03.onrender.com/

**GitHub:** https://github.com/beep-boop-ship-it/jan-setu

**Team:** beep-boop-ship-it
