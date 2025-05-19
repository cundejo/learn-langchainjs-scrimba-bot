import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

/**
 * Convert a question to a standalone question.
 */

const openAIApiKey = process.env.OPENAI_API_KEY;

const llm = new ChatOpenAI({ openAIApiKey });

const standaloneQuestionTemplate =
  "Given the following question, convert it to a standalone question: {question}";

const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate,
);

const standaloneQuestionChain = standaloneQuestionPrompt.pipe(llm);

const response = await standaloneQuestionChain.invoke({
  question:
    "I suppose you have videos in the course, because I learn better when I can see things.",
});

console.log(response);
