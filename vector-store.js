import "dotenv/config";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { OllamaEmbeddings } from "@langchain/ollama";
import { createClient } from "@supabase/supabase-js";

export const vectorStore = () => {
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
  });

  const supabaseClient = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_API_KEY,
  );

  const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
  });

  return {
    get: () => vectorStore,
    search: async (query, numberOfResults = 2) =>
      await vectorStore.similaritySearch(query, numberOfResults),
  };
};

//Testing the function
// const documents = await vectorStore().search(
//   "What is the price of the course?",
//   1,
// );
// for (const doc of documents) {
//   console.log(`* ${doc.pageContent} [${JSON.stringify(doc.metadata, null)}]`);
// }
