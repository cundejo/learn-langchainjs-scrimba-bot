import "dotenv/config";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { vectorStore } from "./vector-store.js";
import { models } from "./models.js";

/**
 * Convert a question to a standalone question.
 */

const vectorStoreInstance = vectorStore().get();

const retriever = vectorStoreInstance.asRetriever(1);

const llm = models.openai();

const standaloneQuestionTemplate =
  "Given the following question, convert it to a standalone question: {question}";

const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate,
);

const chain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser())
  .pipe(retriever);

const response = await chain.invoke({
  question:
    "I'm novice in programming and I learn better when I can visualize things instead of reading. Is this course for me?",
});

console.log(response);
