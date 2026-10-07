FROM node:22-alpine AS build
WORKDIR /workspace/web
COPY web/package*.json ./
RUN npm ci
COPY web .
RUN npm run build

FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /workspace/web/dist /usr/share/nginx/html
EXPOSE 80
