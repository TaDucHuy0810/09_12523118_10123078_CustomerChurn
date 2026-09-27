# Frontend

Giao diện web nằm ở thư mục [site](site/), gồm form nhập dữ liệu
khách hàng và thẻ kết quả xác suất rời mạng.

## Chạy bằng Docker

```powershell
docker compose up --build frontend
```

Nginx phục vụ static files và reverse proxy `/api/` đến backend trong Docker
network. Mở `http://localhost:3000` để sử dụng.
