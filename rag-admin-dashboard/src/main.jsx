import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

import { validateStructure } from "./dev/StructureValidator";

(async () => {
  await validateStructure();

  ReactDOM.createRoot(document.getElementById("root")).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
})();