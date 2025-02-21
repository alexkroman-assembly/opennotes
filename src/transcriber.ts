#!/usr/bin/env ts-node
import { Command } from "commander";
import fs from "fs";
import path from "path";
import { exec } from "child_process";
import { AssemblyAI } from "assemblyai";
import dotenv from "dotenv";
import _ from "lodash";
import { PROMPT } from "./prompt";

dotenv.config();

async function callAssemblyForTranscription(filePath: string, prompt: string) {
  const API_KEY = process.env.ASSEMBLYAI_API_KEY;

  if (!API_KEY) {
    throw new Error(
      "Missing AssemblyAI API key. Set ASSEMBLYAI_API_KEY in .env"
    );
  }
  const client = new AssemblyAI({ apiKey: API_KEY });

  console.log(`Transcribing ${filePath} with AssemblyAI...`);

  const fileUrl = await client.files.upload(filePath);
  console.log(`Uploaded File URL: ${fileUrl}`);

  const transcript = await client.transcripts.transcribe({
    audio: fileUrl,
    summarization: true,
    summary_model: "conversational",
    speaker_labels: true,
    summary_type: "bullets_verbose",
  });

  if (transcript.status === "completed") {
    const { response } = await client.lemur.task({
      transcript_ids: [transcript.id],
      prompt,
      final_model: "anthropic/claude-3-5-sonnet",
    });
    return { transcript, response };
  }

  return { transcript };
}

async function callWhisperForTranscription(filePath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const baseName = path.basename(filePath, path.extname(filePath));
    const tempOutputDir = "/tmp";
    const cmd = `whisper "${filePath}" --model small --output_format txt --output_dir ${tempOutputDir}`;
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error("Error running whisper:", error);
        return reject(error);
      }
      const outputFilePath = path.join(tempOutputDir, `${baseName}.txt`);
      fs.readFile(outputFilePath, "utf8", (readErr, data) => {
        if (readErr) {
          return reject(readErr);
        }
        fs.unlink(outputFilePath, (unlinkErr) => {
          if (unlinkErr) {
            console.error(
              "Failed to delete temporary whisper output file:",
              unlinkErr
            );
          }
          resolve(data);
        });
      });
    });
  });
}

async function callWhisperCliForTranscription(
  filePath: string
): Promise<string> {
  return new Promise((resolve, reject) => {
    const WHISPER_CPP = process.env.WHISPER_CPP;
    if (!WHISPER_CPP) {
      return reject(
        new Error("Missing Whisper CPP path. Set WHISPER_CPP in .env")
      );
    }

    const baseName = path.basename(filePath, path.extname(filePath));
    const tempOutputDir = "/transcriptions";
    const outputFilePath = path.join(tempOutputDir, `${baseName}.txt`);
    const cmd = `${WHISPER_CPP}/build/bin/whisper-cli --model ${WHISPER_CPP}/models/ggml-base.en.bin --output-txt --output-file ${outputFilePath} ${filePath}`;

    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error("Error running whisper-cli:", error);
        return reject(error);
      }
      return resolve(stdout);
    });
  });
}

async function transcribeFile(
  filePath: string,
  transcriptionsDir: string,
  backend: string
) {
  if (!fs.existsSync(filePath)) {
    console.error(`File ${filePath} does not exist.`);
    return;
  }

  if (!fs.existsSync(transcriptionsDir)) {
    fs.mkdirSync(transcriptionsDir, { recursive: true });
  }

  const baseName = path.basename(filePath, path.extname(filePath));
  const transcriptionFilePath = path.join(transcriptionsDir, `${baseName}.txt`);

  if (fs.existsSync(transcriptionFilePath)) {
    console.log(`Skipping ${filePath}, transcription file already exists.`);
    return;
  }

  try {
    let transcriptionResult: string = "";
    if (backend === "assembly") {
      const { transcript, response } = await callAssemblyForTranscription(
        filePath,
        PROMPT
      );
      if (transcript.status === "completed" && transcript.utterances) {
        const formattedTranscript = _.map(
          transcript.utterances,
          (utterance: any) => `Speaker ${utterance.speaker}: ${utterance.text}`
        ).join("\n");

        transcriptionResult = `LLM Transcript Summary:\n ${response}\n\nTranscript Summary:\n${transcript.summary}\n\n${formattedTranscript}`;
      } else {
        transcriptionResult = "";
      }
    } else if (backend === "whisper") {
      console.log(`Transcribing ${filePath} using Whisper...`);
      transcriptionResult = await callWhisperForTranscription(filePath);
    } else if (backend === "whisper-cli") {
      console.log(`Transcribing ${filePath} using whisper-cli...`);
      transcriptionResult = await callWhisperCliForTranscription(filePath);
    } else {
      console.error(`Unknown backend: ${backend}`);
      return;
    }

    fs.writeFileSync(transcriptionFilePath, transcriptionResult);
    console.log(`Transcript saved to ${transcriptionFilePath}`);
  } catch (error) {
    console.error(`Error transcribing ${filePath}:`, error);
  }
}

async function transcribeAudioFiles(
  recordingsDir: string,
  transcriptionsDir: string,
  backend: string
) {
  if (!fs.existsSync(recordingsDir)) {
    console.error(`Recordings directory ${recordingsDir} does not exist.`);
    return;
  }

  const files = fs.readdirSync(recordingsDir);
  const mp3Files = _.filter(
    files,
    (file) => path.extname(file).toLowerCase() === ".wav"
  );

  // Transcribe all files concurrently
  await Promise.all(
    mp3Files.map((mp3File) =>
      transcribeFile(
        path.join(recordingsDir, mp3File),
        transcriptionsDir,
        backend
      )
    )
  );

  console.log("All possible transcriptions are complete.");
}

async function transcribeSingleFile(
  file: string,
  transcriptionsDir: string,
  backend: string
) {
  await transcribeFile(file, transcriptionsDir, backend);
}

const program = new Command();

program
  .name("transcribe")
  .description(
    "Transcribe audio files using AssemblyAI, Whisper, or whisper-cli (whisper.cpp)"
  )
  .version("1.0.0")
  .option(
    "-r, --recordings <dir>",
    "Directory containing audio recordings",
    "recordings"
  )
  .option(
    "-t, --transcriptions <dir>",
    "Directory to save transcriptions",
    "transcriptions"
  )
  .option("-f, --file <file>", "Single audio file to transcribe")
  .option(
    "-b, --backend <backend>",
    "Transcription backend: whisper, whisper-cli, or assembly",
    "whisper"
  )
  .action(async (options) => {
    try {
      if (options.file) {
        await transcribeSingleFile(
          options.file,
          options.transcriptions,
          options.backend
        );
      } else {
        await transcribeAudioFiles(
          options.recordings,
          options.transcriptions,
          options.backend
        );
      }
      console.log("Done!");
      process.exit(0);
    } catch (err) {
      console.error(err);
      process.exit(1);
    }
  });

program.parse(process.argv);
