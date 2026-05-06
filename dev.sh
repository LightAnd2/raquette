#!/bin/bash
# Start both servers for local development

echo "Starting Raquette..."

# Backend
(cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000) &
BACKEND_PID=$!

# Frontend
(cd frontend && npm run dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
