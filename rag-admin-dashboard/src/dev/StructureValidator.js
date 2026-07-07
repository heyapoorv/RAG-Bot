const requiredStructure = [
  "src/api/client.js",
  "src/components/Chat/ChatWindow.jsx",
  "src/components/Chat/InputBox.jsx",
  "src/components/Chat/Message.jsx",
  "src/components/Upload/UploadAttachment.jsx",
  "src/pages/ChatPage.jsx",
  "src/routes/AppRouter.jsx",
];

export async function validateStructure() {
  const missing = [];

  for (const path of requiredStructure) {
    try {
      await import(`../${path}`);
    } catch (e) {
      console.error(`Failed to import ${path}:`, e);
      missing.push(path);
    }
  }

  if (missing.length > 0) {
    console.error("❌ Missing files:", missing);
  } else {
    console.log("✅ All required files present");
  }
}