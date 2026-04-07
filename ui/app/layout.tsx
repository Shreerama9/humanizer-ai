import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Humanizer AI — SEO Content Studio",
  description: "Generate & humanize SEO content that bypasses AI detection",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-50 text-slate-900 antialiased">
        <header className="bg-slate-900 text-white px-8 py-4 flex items-center gap-3">
          <span className="text-xl">✦</span>
          <span className="font-bold text-lg tracking-tight">Humanizer AI</span>
          <span className="text-slate-400 text-sm ml-2">SEO Content Studio</span>
        </header>
        <main className="max-w-5xl mx-auto p-8">{children}</main>
      </body>
    </html>
  );
}
