const API_URL = "http://127.0.0.1:8000"; // backend address

// Test GET request
export async function getHealth() {
  const response = await fetch(`${API_URL}/`);
  return response.json();
}

// File upload
export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  return response.json();
}
