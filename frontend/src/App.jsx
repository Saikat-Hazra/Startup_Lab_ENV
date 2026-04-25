import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard";
import Graph from "./components/Graph";
import { getState, stepSimulation } from "./api";

export default function App() {
  const [statePayload, setStatePayload] = useState({
    state: null,
    actions: [],
    rewards: [],
    insights: [],
  });
  const [rewardHistory, setRewardHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      const data = await getState();
      setStatePayload(data);
    };
    load();
  }, []);

  const runStep = async () => {
    setLoading(true);
    try {
      const data = await stepSimulation();
      setStatePayload(data);

      const rewardSum = (data.rewards || []).reduce((acc, value) => acc + value, 0);
      setRewardHistory((prev) => [
        ...prev,
        { step: data.state?.step ?? prev.length + 1, reward: Number(rewardSum.toFixed(3)) },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-5xl mx-auto space-y-5">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Self-Improving Startup Lab</h1>
          <button
            onClick={runStep}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Running..." : "Run Step"}
          </button>
        </header>

        <Dashboard
          state={statePayload.state}
          actions={statePayload.actions}
          rewards={statePayload.rewards}
          insights={statePayload.insights}
        />

        <Graph rewardHistory={rewardHistory} />
      </div>
    </main>
  );
}
