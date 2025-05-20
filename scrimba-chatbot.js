import "dotenv/config";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { models } from "./models.js";
import { vectorStore } from "./vector-store.js";

/**
 * The Scrimba Chatbot does the following:
 * 1. Receives a question from the user, saved in the const USER_QUESTION
 * 2. Converts the question to a standalone question
 * 3. Retrieves the most relevant documents from the vector store based on the standalone question.
 *    You can control the number of documents retrieved by changing VECTOR_STORE_K
 * 4. Past to the LLM the original user question, the standalone question and
 *    the retrieved documents to get a proper answer
 */

const USER_QUESTION = "I like to learn watching videos. Is this course for me?";

const VECTOR_STORE_K = 1;

const llm = models.openai();

// Convert a question to a standalone question.
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  `Given the following question, convert it to a standalone question: "{question}"`,
);

const standaloneQuestionChain = RunnableSequence.from([
  standaloneQuestionPrompt,
  llm,
  new StringOutputParser(),
]);

// Retrieves the most relevant documents from the vector store based on the standalone question.
const vectorStoreInstance = vectorStore().get();

const retriever = vectorStoreInstance.asRetriever(VECTOR_STORE_K);

const documentsChain = RunnableSequence.from([
  (input) => input.standaloneQuestion,
  retriever,
  (input) => input.map((doc) => doc.pageContent).join("\n\n"),
]);

// Past to the LLM the original user question, and the retrieved documents as
//  context to get a proper answer
const answerPrompt = PromptTemplate.fromTemplate(`
You are a helpful and enthusiastic support bot who can answer a given question 
about Scrimba based on the context provided. You'll always try to find the 
answer in the context, if you can't find it, say 
"I'm sorry, I don't know the answer to that.", and direct the questioner to 
email help@scrimba.com. Don't try to make up an answer. 
Question: "{question}"
Context: "{context}"
`);

const answerChain = RunnableSequence.from([
  {
    question: (input) => input.originalQuestion,
    context: (input) => input.context,
  },
  answerPrompt,
  llm,
  new StringOutputParser(),
]);

// Putting it all together

const chain = RunnableSequence.from([
  // logger,
  {
    standaloneQuestion: standaloneQuestionChain,
    originalQuestion: (input) => input.question,
  },
  // logger,
  {
    context: documentsChain,
    originalQuestion: (input) => input.originalQuestion,
    standaloneQuestion: (input) => input.standaloneQuestion,
  },
  // logger,
  answerChain,
]);

const response = await chain.invoke({
  question: USER_QUESTION,
});

console.log(response);
