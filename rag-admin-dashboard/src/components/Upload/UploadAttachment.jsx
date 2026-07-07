export default function UploadAttachment({ fileRef }) {
  return (
    <label className="upload-btn">
      +
      <input type="file" ref={fileRef} hidden />
    </label>
  );
}