import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = { title: "RAGLab — RAG Evaluation Workbench", description: "Measure, debug, and improve retrieval augmented generation systems." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
