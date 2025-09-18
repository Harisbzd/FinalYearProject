import React, { useState } from "react";
import questions from "../data/questions";

interface ChatBoxProps {
  onClose: () => void;
}

interface AnswerEntry {
  question: string;
  answer: string;
}

export default function ChatBox({ onClose }: ChatBoxProps): React.JSX.Element {
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [answers, setAnswers] = useState<AnswerEntry[]>([]);
  const [input, setInput] = useState<string>("");

  const handleSend = (): void => {
    if (!input.trim()) return;

    const newAnswer: AnswerEntry = { question: questions[currentIndex], answer: input };
    const updatedAnswers: AnswerEntry[] = [...answers, newAnswer];

    setAnswers(updatedAnswers);
    setInput("");

    if (currentIndex < questions.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      setCurrentIndex(-1);

      fetch("/submit-answers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updatedAnswers),
      })
        .then((res) => res.json())
        .catch((err) => console.error("Error submitting answers:", err));
    }
  };

  return (
    <div className="p-4 bg-gray-100 h-96 rounded-lg flex flex-col">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-bold text-gray-800">🩺 Diabetes Assistant</h2>
        <button onClick={onClose}>✖️</button>
      </div>

      <div className="flex-1 overflow-y-auto bg-white rounded p-2 mb-2">
        {answers.map((entry, index) => (
          <div key={index} className="mb-2">
            <div className="text-sm text-gray-700 mb-1">🤖 {entry.question}</div>
            <div className="ml-4 text-sm text-blue-600">🧑 {entry.answer}</div>
          </div>
        ))}

        {currentIndex >= 0 && (
          <div className="text-sm text-gray-700 mt-4">🤖 {questions[currentIndex]}</div>
        )}

        {currentIndex === -1 && (
          <div className="text-green-600 font-semibold mt-4">
            ✅ Thank you for your responses!
          </div>
        )}
      </div>

      {currentIndex >= 0 && (
        <div className="flex mt-2">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-l px-2 py-1 text-sm"
            placeholder="Type your answer..."
            value={input}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInput(e.target.value)}
            onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && handleSend()}
          />
          <button
            className="bg-blue-600 text-white px-3 py-1 text-sm rounded-r hover:bg-blue-700"
            onClick={handleSend}
          >
            Send
          </button>
        </div>
      )}
    </div>
  );
} 