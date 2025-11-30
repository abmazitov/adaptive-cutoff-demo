# Deployment Guide

This guide walks you through deploying the Adaptive Cutoff Dashboard to various hosting platforms.

## Option 1: Render (Recommended)

Render offers a free tier and is the easiest option for Python web applications.

### Steps:

1. **Initialize Git and Push to GitHub**
   ```bash
   cd /Users/arslanmazitov/projects/adaptive-cutoff-demo
   git init
   git add .
   git commit -m "Initial commit: Adaptive Cutoff Dashboard"
   ```

2. **Create a GitHub Repository**
   - Go to GitHub and create a new repository
   - Push your code:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

3. **Deploy to Render**
   - Go to [render.com](https://render.com) and sign in with GitHub
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` configuration
   - Click "Create Web Service"

4. **Wait for Build**
   - Render will install dependencies and start your app
   - You'll get a public URL like `https://adaptive-cutoff-demo.onrender.com`

### Notes:
- Free tier: Your app will spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- No credit card required for free tier

## Option 2: Railway

Railway offers a generous free tier with automatic deployments.

### Steps:

1. **Push to GitHub** (same as Render steps 1-2)

2. **Deploy to Railway**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Choose "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the Python app and use the `Procfile`

3. **Wait for Deployment**
   - Railway will provide a public URL

## Option 3: Google Cloud Run

For production workloads with auto-scaling.

### Steps:

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD ["python", "-m", "adaptive_cutoff_demo.dashboard"]
   ```

2. **Build and Deploy**
   ```bash
   gcloud run deploy adaptive-cutoff-demo \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Option 4: PythonAnywhere

Good for simple deployments with a web interface.

### Steps:

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload your code via Git or web interface
3. Create a new web app with manual configuration
4. Set the WSGI file to use your Dash app
5. Install dependencies via Bash console

## Troubleshooting

### Build Failures

**Issue**: Installation of metatrain from GitHub fails
- **Solution**: Ensure the `adaptive-cutoff` branch exists and is public
- Check that all dependencies are available for your Python version

**Issue**: PyTorch installation fails
- **Solution**: You may need to specify CPU-only PyTorch in requirements.txt:
  ```
  torch --index-url https://download.pytorch.org/whl/cpu
  ```

### Runtime Errors

**Issue**: Port binding errors
- **Solution**: The app uses PORT environment variable, which platforms set automatically
- For local testing, set PORT manually: `PORT=8050 python -m adaptive_cutoff_demo.dashboard`

**Issue**: Memory limits exceeded
- **Solution**: Free tiers have memory limits (usually 512MB)
- Consider upgrading to paid tier or optimizing memory usage

## Environment Variables

All platforms support these environment variables:

- `PORT`: Port number (auto-set by platform)
- `HOST`: Host to bind to (defaults to 0.0.0.0)

## Monitoring

After deployment:
- Check logs on your platform's dashboard
- Monitor memory and CPU usage
- Set up uptime monitoring (e.g., UptimeRobot)

## Custom Domain

Most platforms allow custom domains:
- Render: Settings → Custom Domain
- Railway: Settings → Domains
- Google Cloud Run: Domain mappings

## Updating

To update your deployed app:
1. Make changes locally
2. Commit and push to GitHub
3. Platform will auto-deploy (if enabled)

Or trigger manual deployment from platform dashboard.
