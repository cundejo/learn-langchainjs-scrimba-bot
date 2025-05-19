import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

export const models = {
  openai: () =>
    new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY,
      model: "gpt-4.1-nano-2025-04-14",
      temperature: 0,
    }),
};
