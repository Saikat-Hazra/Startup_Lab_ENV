import axios from "axios";

const client = axios.create({
  baseURL: "http://localhost:8000",
});

export async function getState() {
  const { data } = await client.get("/state");
  return data;
}

export async function stepSimulation() {
  const { data } = await client.post("/step", {});
  return data;
}
