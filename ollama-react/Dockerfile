# Stage 1.  Compile and build angular codebase
# FROM node:lts-alpine AS build
FROM node:18-bullseye-slim AS build

# Set the working directory
WORKDIR /usr/src/app

# copy package.json
COPY package.json ./

# install dependencies
RUN npm install

COPY . .

# build application
RUN npm run build

# Stage 2.  Setup nginx.
FROM nginx:1.24.0-bullseye AS serve

# copy the build output to replace the default nginx contents
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /usr/src/app/build /var/www/html/ollama-pi

# EXPORT port 80
EXPOSE 80

STOPSIGNAL SIGTERM