import React, { useState } from "react";
import ChatBox from "./ChatBox";

// ChatBoxProps interface is used by ChatBox component

export default function ChatWidget(): React.JSX.Element {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  return (
    <>
      {/* Floating Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <button
          className="bg-blue-300 text-white rounded-full p-4 shadow-lg hover:bg-blue-700 transition"
          onClick={() => setIsOpen(!isOpen)}
        >
          Need Help?
        </button>
      </div>

      {/* ChatBox */}
      {isOpen && (
        <div className="fixed bottom-20 right-6 w-80 bg-gray-100 rounded-lg shadow-xl z-50">
          <ChatBox onClose={() => setIsOpen(false)} />
        </div>
      )}
    </>
  );
} 