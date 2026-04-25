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
    reasonings: [],
    commentary: "",
  });
  const [rewardHistory, setRewardHistory] = useState([{ step: 0, reward: 0 }]);
  const [loading, setLoading] = useState(false);
  const [strategyChange, setStrategyChange] = useState(false);
  const [prevActions, setPrevActions] = useState([]);
  const [mode, setMode] = useState("trained"); // "baseline" or "trained"

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getState();
        setStatePayload(data);
      } catch (error) {
        console.error("Error loading initial state:", error);
      }
    };
    load();
  }, []);

  const runStep = async () => {
    setLoading(true);
    try {
      const data = await stepSimulation(mode);
      setStatePayload(data);

      // Detect strategy change
      const actionsChanged = JSON.stringify(data.actions) !== JSON.stringify(prevActions);
      setStrategyChange(actionsChanged);
      setPrevActions(data.actions);

      const rewardSum = (data.rewards || []).reduce((acc, value) => acc + value, 0);
      setRewardHistory((prev) => [
        ...prev,
        { step: data.state?.step ?? prev.length + 1, reward: Number(rewardSum.toFixed(3)) },
      ]);
    } catch (error) {
      console.error("Error running simulation step:", error);
      // Could add user notification here
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-5xl mx-auto space-y-5">
        <header className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Self-Improving Startup Lab</h1>
          <div className="flex items-center space-x-4">
            {strategyChange && (
              <div className="text-yellow-600 font-semibold">
                ⚡ Strategy Shift Detected
              </div>
            )}
            <button
              onClick={() => setMode(mode === "trained" ? "baseline" : "trained")}
              className="px-4 py-2 rounded-lg bg-gray-600 text-white hover:bg-gray-700"
            >
              👉 Show {mode === "trained" ? "Untrained" : "Trained"}
            </button>
            <button
              onClick={runStep}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? "Running..." : "Run Step"}
            </button>
          </div>
        </header>

        <Dashboard
          state={statePayload.state}
          actions={statePayload.actions}
          rewards={statePayload.rewards}
          insights={statePayload.insights}
          reasonings={statePayload.reasonings}
          commentary={statePayload.commentary}
        />

        <Graph rewardHistory={rewardHistory} />
      </div>
    </main>
  );
}
