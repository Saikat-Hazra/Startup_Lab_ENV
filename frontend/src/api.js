import axios from "axios";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? "http://127.0.0.1:8002" : "");

const client = axios.create({
  baseURL: API_BASE_URL,
});

export async function getState() {
  const { data } = await client.get("/state");
  return data;
}

export async function stepSimulation(mode = "trained") {
  const { data } = await client.post("/step", { mode });
  return data;
}
