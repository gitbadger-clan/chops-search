+++
title = "Rusty puppets, Websockets and Voyeurism (part II): Driving Chromium in Docker with a Window"
date = 2025-11-17
draft = false
[taxonomies]
tags = ["rust", "docker", "containers", "chrome", "chromium", "arm64"]
+++

## TL;DR

You swapped Chrome → Chromium for better arm64 support, strapped on VNC + noVNC to watch the chaos, made Alpine optional to chase size gains, and pimped a Makefile so everything feels like a dead man's switch.

## The Backstory

In some of my previous experiments I have used browser automation with Chrome to extract info from pages, usually via a wrapper around CDP. CDP has a pretty neat and quite huge API for driving Chrome and Chromium based browsers. Instead of using an existing wrapper, I built my own in Rust with `tungstenite` to communicate over WebSocket.

![Hero image](rusty-puppets-docker-chrome.png)

## Why?

Ideally what I want is a server style primitive component that is able to speak CDP over a websocket without having to babysit a local browser instance. Normally the way you run most of the chrome automations is by running the process and if you want some transparency into what is happening you flip `--headless` flag to false, but this is not what you want, the automation is outside the docker container, the container only holds the browser with the remote debugging port exposed. Something else that tripped me up was that there were no Chrome repos with arm64 builds that i could run on my M1 mac, so I switched to Chromium that supports arm64 better.

Yes, "headless" is efficient; no, I don't trust it until I can see it wiggle, that is why an extra feature that felt right was having VNC enabled on one of the container variants.

## What?

Components (lego brick style)

- Chromium (not Chrome) – better support for arm64 builds, CDP at 9222.
- Socat - port forwarding for CDP, since binding to 0.0.0.0 did not work well with Chromium.
- Xvfb + lightweight WM (fluxbox) – fake display for the VNC stream.
- VNC server (x11vnc) – the eyeballs.
- noVNC + websockify – view it in the browser at :6080.
- supervisord – herd cats (multiple daemons).
- Alpine (optional) – minimal base; trade-offs: fonts, glibc shims, weird edges.
- Rust client – talks to <http://container:9222/json>

## Architecture

```less
+------------------ Docker Container --------------------+
|                                                        |
|  [Xvfb] --- [WM] --- [VNC Server] --- [noVNC]          |
|                         ^             (HTTP 8080)      |
|                         |                              |
|                    screen:0                            |
|                                                        |
|  [Chromium --headless=new --remote-debugging-port=9222]|
|                                         (WS:9222)      |
|                                ^                       |
|                                |                       |
|                            [Socat proxy]               |
+--------------------------------------------------------+

 Outside:
   Rust (chromiumoxide/others) --> http://host:<exposed-socat-port>
   Human -> http://host:6080 (noVNC)

```

## How?

1. Image strategy (two by two):

Essentially what I wanted was efficiency at the container level and at the browser level. I applied that in practice by using `alpine` for smaller image size and running headless for better runtime efficiency. However running blind is kinda hard to introspect so I added two additional images with VNC support and those came with both `alpine` and `debian` flavors.

- Debian/Ubuntu base: fewer papercuts, bigger image, fastest to "it works."
- Alpine base: smallest, but bring your own fonts, codecs, and glibc cuddles.

> Pro tip: start Debian for DX, ship Alpine once you tame fonts/codecs.

2. Dockerfile (there is a common base between `debian` and `alpine`)

```dockerfile
#....
# Default ports
ENV CHROME_PORT=9222
ENV SOCAT_PORT=9224
ENV VNC_PORT=5900
ENV NOVNC_PORT=6080

#....packages common to both debian and alpine
  vim \ # want to edit stuff?
  chromium \ # duh
  socat \ # port forwarding for the 9222 remote debugging port
  curl \ # healthcheck and manual testing
  net-tools \ # netstat useful for listing open ports
  iproute2 \ # ss useful for listing open ports
  ca-certificates \
  procps \


#.... Install VNC and GUI components
  supervisor \
  xvfb \
  x11vnc \
  websockify \
  fluxbox \
  git \

#....

# Install noVNC from source (most reliable method)
RUN cd /opt && \
  git clone --depth 1 https://github.com/novnc/noVNC.git && \
  cd noVNC && \
  ln -s vnc.html index.html

#....

# Create necessary directories with proper permissions
RUN mkdir -p /home/chrome/data && \
  mkdir -p /var/log/supervisor && \
  chown -R chrome:chrome /home/chrome && \
  chown -R chrome:chrome /var/log/supervisor


# Copy startup script
ARG STEALTH=basic
COPY ${STEALTH}.sh /usr/local/bin/start-chrome.sh
RUN chmod +x /usr/local/bin/start-chrome.sh && \
  chown chrome:chrome /usr/local/bin/start-chrome.sh

#....


EXPOSE ${SOCAT_PORT} ${VNC_PORT} ${NOVNC_PORT}

CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

3. Dockerfile (lightweight Debian to get a more frictionless experience):

```dockerfile
FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
  #....
  fonts-liberation \
  fonts-noto-color-emoji \
  fonts-roboto \
  fonts-noto \
  libasound2 \
  libatk-bridge2.0-0 \
  libatk1.0-0 \
  libatspi2.0-0 \
  libcups2 \
  libdbus-1-3 \
  libdrm2 \
  libgbm1 \
  libgtk-3-0 \
  libnspr4 \
  libnss3 \
  libwayland-client0 \
  libxcomposite1 \
  libxdamage1 \
  libxfixes3 \
  libxkbcommon0 \
  libxrandr2 \
  xdg-utils \
  && rm -rf /var/lib/apt/lists/*

# Install VNC and GUI components
RUN apt-get update && apt-get install -y --no-install-recommends \
  gnupg \
  #....
  python3 \
  python3-numpy \
  && rm -rf /var/lib/apt/lists/*

#....

# Create a non-root user
RUN useradd -m -s /bin/bash chrome

#....

# Copy supervisord config
COPY supervisord.conf.debian /etc/supervisor/conf.d/supervisord.conf

#....
```

4. The `alpine` version, similar but not quite the same, some commands are different and some of the package names and dependencies differ too:

```dockerfile
FROM alpine:3.19

#....

# Install base dependencies and Chromium
RUN apk add --no-cache \
  # Core utilities
  bash \ # using some custom scripts to launch the browser
  # ....
  xdpyinfo \ # x11 utilities that are not included by default
  xauth \
  xprop \
  xwininfo \
  # Fonts
  font-liberation \
  font-noto \
  font-noto-emoji \
  font-noto-cjk \
  # Chromium dependencies
  libstdc++ \
  harfbuzz \
  nss \
  freetype \
  ttf-freefont \
  wqy-zenhei \
  # Audio/Video libraries
  alsa-lib \
  at-spi2-core \
  cups-libs \
  dbus-libs \
  libdrm \
  mesa-gbm \
  libxcomposite \
  libxdamage \
  libxfixes \
  libxkbcommon \
  libxrandr \
  wayland-libs-client \
  # X11 libraries
  libx11 \
  libxext \
  libxrender \
  libxtst \
  libxi

# Install VNC and GUI components
RUN apk add --no-cache \
  #....
  python3 \
  py3-numpy \
  py3-pip

#....

# Create a non-root user (Alpine uses adduser instead of useradd)
RUN adduser -D -s /bin/sh chrome

#....

# Copy supervisord config
COPY supervisord.conf.alpine /etc/supervisor/conf.d/supervisord.conf

#....
```

5. In order to manage the multiple containers `k8s` would be overkill so I used `docker-compose` instead. I like it for simple experiments when I have to do quick and incremental iteration for testing my containers. There is less of a need for cleaning up things after and less of a chance to make mistakes with `docker run` commands.

```yaml
name: chrome-cdp

x-chrome-common: &chrome-common
  image: chrome-cdp:${DISTRO:-debian}-${MODE:-headless}-stealth-${STEALTH:-basic}
  container_name: chrome-${DISTRO:-debian}-${MODE:-headless}-stealth-${STEALTH:-basic}
  networks: [cdpnet]
  shm_size: "${HEADLESS_SHM_SIZE}"
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-fsS", "http://0.0.0.0:${SOCAT_PORT}/json/version"]
    interval: 10s
    timeout: 3s
    retries: 5
    start_period: 15s

services:
  chrome-headless:
    <<: *chrome-common
    build:
      context: ./headless
      dockerfile: Dockerfile.${DISTRO:-debian}
      args:
        STEALTH: ${STEALTH:-basic}
    ports:
      - "${CDP_HOST_PORT}:${SOCAT_PORT}"

  chrome-gui:
    <<: *chrome-common
    build:
      context: ./gui
      dockerfile: Dockerfile.${DISTRO:-debian}
      args:
        STEALTH: ${STEALTH:-basic}
    ports:
      - "${CDP_HOST_PORT}:${SOCAT_PORT}"
      - "${VNC_PORT}:5900"
      - "${NOVNC_PORT}:6080"

networks:
  cdpnet:
    driver: bridge
```

as you can see the `docker-compose` file is slightly streamlined by using YAML anchors and aliases to avoid repetition between the two services. But this is not the most interesting DX improvements I added.

6. The pimped out `Makefile`. The big deal with this one is that I got everything working almost like an extension of make, where previously I would just be creating targets to wrap `docker-compose` now I added a bunch of more advanced params to allow parametrization of each target instead of doing different names for each variant.

```Makefile
SHELL := /bin/bash
ENV_FILE ?= .env
STEALTH ?= basic
MODE ?= headless
DISTRO ?= debian

# Container name (adjust to match your docker-compose service name)
CONTAINER_NAME ?= chromium
COMPOSE_FILE ?= docker-compose.yml

.PHONY: health list-containers ps stats stats-all stats-live top \
    ports ports-all ports-detailed logs logs-chrome logs-chrome-live \
    shell rebuild up down chrome-version chrome-tabs chrome-health verify-chrome-flags \
    restart-all stop-all wsurl

# ============================================
# Configuration
# ============================================
CHROME_IMAGE_PREFIX := chrome-cdp
CHROME_IMAGES := \
 $(CHROME_IMAGE_PREFIX):debian-headless-stealth-basic \
 $(CHROME_IMAGE_PREFIX):debian-headless-stealth-advanced \
 $(CHROME_IMAGE_PREFIX):debian-gui-stealth-basic \
 $(CHROME_IMAGE_PREFIX):debian-gui-stealth-advanced \
 $(CHROME_IMAGE_PREFIX):alpine-headless-stealth-basic \
 $(CHROME_IMAGE_PREFIX):alpine-headless-stealth-advanced \
 $(CHROME_IMAGE_PREFIX):alpine-gui-stealth-basic \
 $(CHROME_IMAGE_PREFIX):alpine-gui-stealth-advanced

# Build docker filter arguments
DOCKER_FILTERS := $(foreach img,$(CHROME_IMAGES),--filter "ancestor=$(img)")

# Command to get container
GET_CONTAINER = docker ps $(DOCKER_FILTERS) -q | head -1

# Command to get all containers
GET_ALL_CONTAINERS = docker ps $(DOCKER_FILTERS) -q

# ============================================
# Helper Functions
# ============================================
# Check if container exists and set CONTAINER variable
define require_container
 $(eval CONTAINER := $(shell $(GET_CONTAINER)))
 @if [ -z "$(CONTAINER)" ]; then \
  echo "❌ No $(CHROME_IMAGE_PREFIX) container running"; \
  echo "Available images: $(CHROME_IMAGES)"; \
  exit 1; \
 fi
 @echo "✓ Using container: $(CONTAINER)"
endef

# ============================================
# Targets
# ============================================
help:
 @echo "Chrome Container Management"
 @echo ""
 @echo "Available targets:"
 @echo "  make list-containers    - List all running chrome containers"
 @echo "  make verify-chrome-flags - Verify stealth flags in Chrome"
 @echo "  make stats              - Show container stats"
 @echo "  make stats-all          - Show stats for all chrome containers"
 @echo "  make logs               - Show container logs"
 # @echo "  make exec CMD=<cmd>     - Execute command in container"
 @echo "  make shell              - Open shell in container"

list-containers:
 @echo "Running chrome-cdp containers:"
 @docker ps $(DOCKER_FILTERS) --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

up:
 @echo "Building and starting chromium container in $(MODE) mode with $(DISTRO) (stealth: $(STEALTH))"
 DISTRO=$(DISTRO) MODE=$(MODE) STEALTH=$(STEALTH) docker compose --env-file $(ENV_FILE) up chrome-$(MODE) -d --build

down:
 docker compose --env-file $(ENV_FILE) down

logs:
 $(call require_container)
 @docker logs -f $(CONTAINER)

shell:
 $(call require_container)
 @docker exec -it $(CONTAINER) bash

rebuild:
 DISTRO=$(DISTRO) MODE=$(MODE) STEALTH=$(STEALTH) docker compose --env-file $(ENV_FILE) build --no-cache chrome-$(MODE)

ps:
 docker compose --env-file $(ENV_FILE) ps

health:
 @echo "Headless:" && curl -fsS http://127.0.0.1:$$(grep ^CDP_HOST_PORT $(ENV_FILE) | cut -d= -f2)/json/version | jq -r .Browser || true
 @echo "GUI:" && curl -fsS http://127.0.0.1:$$(grep ^CDP_PORT_GUI $(ENV_FILE) | cut -d= -f2)/json/version | jq -r .Browser || true

# Quick helper to print a page websocketDebuggerUrl (requires jq)
wsurl:
 @curl -s "http://127.0.0.1:$$(grep ^CDP_PORT_HEADLESS $(ENV_FILE) | cut -d= -f2)/json/new?about:blank" | jq -r .webSocketDebuggerUrl
 @curl -s "http://127.0.0.1:$$(grep ^CDP_PORT_GUI $(ENV_FILE) | cut -d= -f2)/json/new?about:blank" | jq -r .webSocketDebuggerUrl


# ============================================
# Chrome-specific commands
# ============================================
chrome-version:
 $(call require_container)
 @docker exec $(CONTAINER) chromium --version

chrome-tabs:
 $(call require_container)
 @CHROME_PORT=$$(docker port $(CONTAINER) | grep -E '9222|9223|9224' | head -1 | awk -F' -> ' '{print $$2}'); \
 if [ -z "$$CHROME_PORT" ]; then \
  echo "❌ Chrome DevTools port not found"; \
  exit 1; \
 fi; \
 echo "Fetching tabs from http://$$CHROME_PORT/json"; \
 curl -s "http://$$CHROME_PORT/json" | jq -r '.[] | "\(.id): \(.title) - \(.url)"'

chrome-health:
 $(call require_container)
 @CHROME_PORT=$$(docker port $(CONTAINER) | grep -E '9222|9223|9224' | head -1 | awk -F' -> ' '{print $$2}'); \
 if [ -z "$$CHROME_PORT" ]; then \
  echo "❌ Chrome DevTools port not found"; \
  exit 1; \
 fi; \
 echo "=== Chrome DevTools Health Check ==="; \
 echo "Endpoint: http://$$CHROME_PORT"; \
 echo ""; \
 if curl -sf "http://$$CHROME_PORT/json/version" >/dev/null 2>&1; then \
  echo "✅ Chrome DevTools is responding"; \
  echo ""; \
  curl -s "http://$$CHROME_PORT/json/version" | jq -r '"Browser: \(.Browser)\nProtocol Version: \(."Protocol-Version")\nUser Agent: \(."User-Agent")"'; \
 else \
  echo "❌ Chrome not responding"; \
  exit 1; \
 fi

verify-chrome-flags:
 $(call require_container)
 @echo "Verifying Chrome flags in container $(CONTAINER)..."
 @docker exec $(CONTAINER) sh -c \
  "cat /proc/\$$(pgrep -o chromium)/cmdline | tr '\0' '\n' | grep -E 'disable-blink-features|user-agent'" \
  && echo "✅ Stealth flags detected" \
  || echo "❌ No stealth flags"

stats:
 $(call require_container)
 @echo "Container stats:"
 @docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}" $(CONTAINER)

stats-all:
 @CONTAINERS=$$($(GET_ALL_CONTAINERS)); \
 if [ -z "$$CONTAINERS" ]; then \
  echo "❌ No $(CHROME_IMAGE_PREFIX) containers running"; \
  exit 1; \
 fi; \
 echo "Stats for all chrome containers:"; \
 docker stats --no-stream --format "table {{.Container}}\t{{.Image}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" $$CONTAINERS

stats-live:
 @CONTAINERS=$$($(GET_ALL_CONTAINERS)); \
 if [ -z "$$CONTAINERS" ]; then \
  echo "❌ No $(CHROME_IMAGE_PREFIX) containers running"; \
  exit 1; \
 fi; \
 docker stats $$CONTAINERS

top:
 @CONTAINER=$$($(GET_CONTAINER))
 @echo "=== Top Processes in Container ==="
 @docker top $$(docker-compose -f $(COMPOSE_FILE) ps -q $$CONTAINER)

# ============================================
# Bulk operations
# ============================================
restart-all:
 @CONTAINERS=$$($(GET_ALL_CONTAINERS)); \
 if [ -z "$$CONTAINERS" ]; then \
  echo "❌ No containers to restart"; \
  exit 1; \
 fi; \
 echo "Restarting all chrome containers..."; \
 docker restart $$CONTAINERS

stop-all:
 @CONTAINERS=$$($(GET_ALL_CONTAINERS)); \
 if [ -z "$$CONTAINERS" ]; then \
  echo "❌ No containers to stop"; \
  exit 1; \
 fi; \
 echo "Stopping all chrome containers..."; \
 docker stop $$CONTAINERS

# Show port mappings for a single container
ports:
 $(call require_container)
 @echo "=== Port Mappings for Container $(CONTAINER) ==="
 @echo ""
 @docker port $(CONTAINER) | awk -F' -> ' '{print $$1 "\t→\t" $$2}' | column -t -s $$'\t'

# Show port mappings for all chrome containers in a table
ports-all:
 @CONTAINERS=$$($(GET_ALL_CONTAINERS)); \
 if [ -z "$$CONTAINERS" ]; then \
  echo "❌ No $(CHROME_IMAGE_PREFIX) containers running"; \
  exit 1; \
 fi; \
 echo "=== Port Mappings for All Chrome Containers ==="; \
 echo ""; \
 printf "%-15s %-40s %-20s %-20s\n" "CONTAINER ID" "IMAGE" "CONTAINER PORT" "HOST BINDING"; \
 printf "%-15s %-40s %-20s %-20s\n" "---------------" "----------------------------------------" "--------------------" "--------------------"; \
 for container in $$CONTAINERS; do \
  IMAGE=$$(docker inspect --format='{{.Config.Image}}' $$container); \
  SHORT_ID=$$(echo $$container | cut -c1-12); \
  docker port $$container | while IFS= read -r line; do \
   CONTAINER_PORT=$$(echo "$$line" | awk -F' -> ' '{print $$1}'); \
   HOST_BINDING=$$(echo "$$line" | awk -F' -> ' '{print $$2}'); \
   printf "%-15s %-40s %-20s %-20s\n" "$$SHORT_ID" "$$IMAGE" "$$CONTAINER_PORT" "$$HOST_BINDING"; \
  done; \
 done

# Detailed port information with service names
ports-detailed:
 $(call require_container)
 @echo "=== Detailed Port Information for Container $(CONTAINER) ==="; \
 echo ""; \
 IMAGE=$$(docker inspect --format='{{.Config.Image}}' $(CONTAINER)); \
 echo "Container: $(CONTAINER)"; \
 echo "Image: $$IMAGE"; \
 echo ""; \
 printf "%-35s %-20s %-25s %-15s\n" "SERVICE" "CONTAINER PORT" "HOST BINDING" "PROTOCOL"; \
 printf "%-35s %-20s %-25s %-15s\n" "----------------------------------" "--------------------" "-------------------------" "---------------"; \
 docker port $(CONTAINER) | while IFS= read -r line; do \
  CONTAINER_PORT=$$(echo "$$line" | awk -F' -> ' '{print $$1}' | cut -d'/' -f1); \
  PROTOCOL=$$(echo "$$line" | awk -F' -> ' '{print $$1}' | cut -d'/' -f2); \
  HOST_BINDING=$$(echo "$$line" | awk -F' -> ' '{print $$2}'); \
  case $$CONTAINER_PORT in \
   9222|9223|9224) SERVICE="Chrome DevTools Socat Proxy" ;; \
   5900) SERVICE="VNC Server" ;; \
   6080) SERVICE="noVNC Web" ;; \
   *) SERVICE="Unknown" ;; \
  esac; \
  printf "%-35s %-20s %-25s %-15s\n" "$$SERVICE" "$$CONTAINER_PORT" "$$HOST_BINDING" "$$PROTOCOL"; \
 done

# View Chrome startup script logs
logs-chrome:
 $(call require_container)
 @echo "=== Supervisor Chrome Program Logs ==="
 @docker exec $(CONTAINER) cat /var/log/supervisor/chrome.log 2>/dev/null || \
  echo "Chrome log file not found"

# Live tail of Chrome logs
logs-chrome-live:
 $(call require_container)
 @echo "=== Live Chrome Logs (Ctrl+C to exit) ==="
 @docker logs -f $(CONTAINER) 2>&1 | grep --line-buffered "CHROMIUM\|chromium\|Chrome"
```

## Perfomance & size notes

Alpine and Debian size are roughly the same with the current layout, so whatever I am doing wrong, needs a bit more investigation. The list of installed packages can probably be trimmed a bit more, maybe I get some extra savings but I don't expect them to be noticeable at small scale.

## Security notes

Haven't added any specific security protocols so don't run this in production without:

- Gating with VPN or bind to 127.0.0.1.
- Add auth to noVNC/websockify or stick it behind a reverse proxy with auth.
- Run as non-root (done above), and keep --no-sandbox only inside containers you control.
- Pin package versions for reproducible builds.

## WTF moments (and fixes)

- Had a bunch a pain with the `chromium` remote debugging port as it would not bind to `0.0.0.0:2222` on the container so I had to add `socat` to forward the port. When trying to connect to the remote debugging port of chromium i was getting a connection reset by peer error. Turns out chromium only binds to localhost inside the container unless you use `socat` to forward the port.
- When the browser was not launching inside the container, I thought I messed up, but, as it turns out, any kind of `--headless` flags should probably be omitted when running with VNC, otherwise the browser will not show up in the VNC session.
- this is a hack and probably you can fix it in a better way, but I am lazy, so 2MB of overhead on alpine is me adding bash. I used bash to start chrome with the wrapper script, add the flags all that jazz.
- in my first attempt I tried to use chrome, but there were some silly issues with arm64 builds, so I switched to chromium which has better support for arm64.

## My conclusion

I will use this setup to automate via CDP. It decouples the browser from the language. I know that this is not ideal as a solo dev but hey it's fun. So I will give it a try.
