# Stage 1: Build the React application
FROM node:18-alpine as build-stage

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
# Set environment variable at build-time (default to proxy /api)
ENV VITE_API_URL=/api
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:1.25-alpine

COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
