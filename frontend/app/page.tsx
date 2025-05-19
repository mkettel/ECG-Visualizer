import Image from "next/image";
import Head from "next/head";
import ECGChart from "@/components/ECGChart";

export default function Home() {
  return (
    <>
      <Head>
        <title>ECG Visualizer</title>
        <meta name="description" content="ECG Rhythm Display" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex flex-col h-screen w-full items-center justify-center">
        <h1 className="text-3xl font-bold mb-2">ECG Rhythm Visualizer</h1>
        <ECGChart />
      </main>
    </>
  );
}
