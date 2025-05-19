import "dotenv/config";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs/promises";
import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OllamaEmbeddings } from "@langchain/ollama";

/**
 * In this script we will:
 * 1. Load the text from the `scrimba-info.txt` file
 * 2. Split the text into chunks
 * 3. Create embeddings for each chunk
 * 4. Store the embeddings in Supabase
 */

const sbApiKey = process.env.SUPABASE_API_KEY;
const sbUrl = process.env.SUPABASE_URL;

const text = await fs.readFile("scrimba-info.txt", "utf-8");

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500, // Default 1000
  chunkOverlap: 50, // Default 200
});

const documents = await splitter.createDocuments([text]);
// console.log(documents);

const client = createClient(sbUrl, sbApiKey);

// Not used Open AI embeddings because is not free
// const embeddings  = new OpenAIEmbeddings({ openAIApiKey, model: "text-embedding-3-small" }),

// Using Ollama embeddings
// Needs ollama installed and running locally, and a model downloaded (`ollama pull nomic-embed-text`)
const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text", // A good free embedding model
  baseUrl: "http://localhost:11434", // Default Ollama URL
});

// console.log("embeddings", embeddings);

await SupabaseVectorStore.fromDocuments(documents, embeddings, {
  client,
  tableName: "documents",
});

console.log("Done");
