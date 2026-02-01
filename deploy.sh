#!/bin/bash
set -e

echo "=============================="
echo "Deploying JobAI Backend"
echo "=============================="

cd /srv/jobai

echo "Pulling latest code..."
git pull origin master

echo "Installing dependencies..."
npm install --production

echo "Restarting backend..."
pm2 restart jobai-backend

echo "Deployment completed successfully!"
echo "=============================="
