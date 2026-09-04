# ShopSense API - the whole service in one portable box.
#
# WHAT A CONTAINER IS
#   An image is a filesystem snapshot plus a command to run. It contains
#   the OS libraries, Python, our dependencies, and our code - everything
#   except the kernel. Run it on your laptop, on CI, or on AWS and you get
#   byte-identically the same environment.
#
#   That is the point: "works on my machine" stops being a sentence anyone
#   has to say. The image IS the machine.
#
# WHAT IS DELIBERATELY *NOT* IN IT
#   Secrets. No API key, no database password, no .env - not in a layer,
#   not in an ENV, nowhere. The image is a public artifact: anyone who can
#   pull it can run `docker history` and read every build step. Config
#   arrives at RUN time (--env-file locally, task environment on AWS).
#   Baking a key into an image is how keys end up on Docker Hub.

FROM python:3.13-slim

# Two Python behaviours that are wrong in a container:
#   PYTHONDONTWRITEBYTECODE - .pyc files in a read-only-ish image are dead
#     weight; the source is never edited after build.
#   PYTHONUNBUFFERED - without it, print() sits in a buffer and your logs
#     appear minutes late, or not at all when the container is killed.
#     This one line is the difference between debuggable and not.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# ---------------------------------------------------------------------------
# LAYER CACHING - why requirements.txt is copied on its own, first.
#
# Docker caches each instruction and reuses it while its inputs are
# unchanged. Dependencies change rarely; our source changes constantly.
# Copying requirements.txt alone means the ~60s pip install is reused on
# every code-only rebuild.
#
# Do it the obvious way instead:
#     COPY . .
#     RUN pip install -r requirements.txt
# ...and editing one comment in api/main.py re-downloads every package.
# Same result, one minute slower, every single build, forever.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the code. Everything in .dockerignore is skipped here - that file is
# what stands between this line and a leaked .env.
COPY . .

# ---------------------------------------------------------------------------
# Run as a NON-ROOT user.
#
# Containers run as root by default. If an attacker ever gets code
# execution inside this one, root makes the rest of their job easy. This
# is the same principle as agent_ro in the database: the process gets the
# privileges its job needs and not one more. Defence in depth means the
# boring layer is also locked.
# ---------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash shopsense \
    && chown -R shopsense:shopsense /app
USER shopsense

# Documentation, not a firewall rule: EXPOSE records the port the app
# listens on by default.
EXPOSE 8000

# Is the app SERVING, or merely still running? A process can be alive and
# wedged. Shell form (no JSON array) so ${PORT} is expanded by /bin/sh.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health')"

# ---------------------------------------------------------------------------
# THE COMMAND, AND THE TWO BUGS IT AVOIDS
#
# 1. ${PORT} MUST BE READ AT RUNTIME.
#    Every container platform (AWS App Runner, ECS, Render, Cloud Run)
#    injects a PORT env var and routes traffic to it. Hardcoding 8000
#    means the platform sends traffic to a port nothing is listening on,
#    and you get "service unhealthy" with a perfectly healthy app.
#
#    But exec form - CMD ["uvicorn", "--port", "$PORT"] - does NOT expand
#    variables: there is no shell involved, so uvicorn would receive the
#    literal string "$PORT" and fail to parse it. Hence `sh -c`.
#
# 2. `exec` MATTERS.
#    Without it, sh stays as PID 1 and uvicorn runs as its child. When the
#    platform sends SIGTERM to stop the container, sh receives it and
#    uvicorn never hears about it - so in-flight requests are killed
#    abruptly after the 10s grace period instead of finishing. `exec`
#    REPLACES the shell with uvicorn, so uvicorn is PID 1 and shuts down
#    gracefully.
#
# --host 0.0.0.0 listens on every interface. The default (127.0.0.1) is
# reachable only from INSIDE the container - the single most common
# "it works locally, the deploy is unreachable".
#
# --proxy-headers tells uvicorn to trust X-Forwarded-* from the platform's
# load balancer, so request URLs and client IPs are the real ones rather
# than the proxy's.
#
# The browser UI ships in this image too, and needs no second process:
# api/main.py mounts frontend/ as static files, so this one command
# serves the page and the API it calls from the same origin.
# ---------------------------------------------------------------------------
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
