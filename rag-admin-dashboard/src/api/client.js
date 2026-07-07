import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 60000,
});

// 🔥 Global response handler
API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API ERROR:", error?.response?.data || error.message);

    return Promise.reject({
      message:
        error?.response?.data?.message ||
        "Server error. Please try again.",
      status: error?.response?.status,
    });
  }
);

// 🔥 Safe wrapper
const safeCall = async (promise) => {
  try {
    const res = await promise;
    return { data: res.data, error: null };
  } catch (err) {
    return { data: null, error: err };
  }
};

export const sendQuery = (payload) =>
  safeCall(API.post("/query", payload));

export const uploadFile = (file, session_id) => {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", session_id);

  return safeCall(API.post("/upload", form));
};

export const getAnalytics = () =>
  safeCall(API.get("/analytics/summary"));

export const getFailures = () =>
  safeCall(API.get("/analytics/failures"));