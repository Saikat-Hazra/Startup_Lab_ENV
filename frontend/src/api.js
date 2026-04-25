import axios from "axios";

const client = axios.create({
  baseURL: "http://localhost:8001",
});

export async function getState() {
  const { data } = await client.get("/state");
  return data;
}

export async function stepSimulation(mode = "trained") {
  const { data } = await client.post("/step", { mode });
  return data;
}
