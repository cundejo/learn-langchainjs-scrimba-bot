import "dotenv/config";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import * as readline from "readline";
import chalk from "chalk";
import { models } from "./models.js";
import { vectorStore } from "./vector-store.js";
import { logger } from "./utils.js";

// Setting the project name for LangSmith
process.env.LANGCHAIN_PROJECT = "scrimba-bot";

const VECTOR_STORE_K = 2;
const conversationHistory = []; // Format: ['User: ', 'AI: ', ...]
const llm = models.openai();

/**
 * Receives a question from the user and does the following:
 * 1. Converts the question to a standalone question
 * 3. Retrieves the most relevant documents from the vector store based on the standalone question.
 *    You can control the number of documents retrieved by changing VECTOR_STORE_K
 * 4. Past to the LLM the original user question, the standalone question and
 *    the retrieved documents to get a proper answer
 */
const answerQuestion = async (question) => {
  const conversationHistoryText = conversationHistory.join("\n");

  // Convert a question to a standalone question.
  const standaloneQuestionPrompt = PromptTemplate.fromTemplate(`
  Given a question and some conversation history (if any), convert the question to a standalone question.
  
  question: "{question}"
  conversation history: "{conversationHistory}"
  `);

  const standaloneQuestionChain = RunnableSequence.from([
    {
      question: () => question,
      conversationHistory: () => conversationHistoryText,
    },
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

  // Past to the LLM the original user question, the conversation history,
  // and the retrieved documents as context to get a proper answer.
  const answerPrompt = PromptTemplate.fromTemplate(`
You are a helpful and enthusiastic support bot who can answer a given question 
about Scrimba based on the context provided and the conversation history. 
Try to find the answer in the context. If the answer is not given in the context, 
find the answer in the conversation history if possible. 
If you really don't know the answer, say "I'm sorry, I don't know the answer to that.",
and direct the questioner to email help@scrimba.com. Don't try to make up an answer. 
Always speak as if you were chatting to a friend. 

question: "{question}"
context: "{context}"
conversation history: "{conversationHistory}"
`);

  const answerChain = RunnableSequence.from([
    {
      question: (input) => input.originalQuestion,
      context: (input) => input.context,
      conversationHistory: () => conversationHistoryText,
    },
    // logger,
    answerPrompt,
    llm,
    new StringOutputParser(),
  ]);

  // Create the final chain and invoke it
  const chain = RunnableSequence.from([
    {
      standaloneQuestion: standaloneQuestionChain,
      originalQuestion: (input) => input.question,
    },
    // logger,
    {
      context: documentsChain,
      originalQuestion: (input) => input.originalQuestion,
    },
    // logger,
    answerChain,
  ]);

  return await chain.invoke({ question });
};

const main = async () => {
  console.log(
    chalk.yellowBright(
      "\n=== Scrimba Chatbot started. Type your questions and press Enter. (Ctrl+C to exit) ===\n",
    ),
  );

  // Create interface for reading from terminal
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  // Function to prompt user and get input
  const getUserInput = () => {
    return new Promise((resolve) => {
      rl.question("> ", (input) => {
        resolve(input);
      });
    });
  };

  try {
    while (true) {
      const userInput = await getUserInput();
      const answer = await answerQuestion(userInput);
      conversationHistory.push(`User: ${userInput}`);
      conversationHistory.push(`Bot: ${answer}`);
      console.log(chalk.green(answer));
    }
  } catch (error) {
    console.error("Error:", error);
  } finally {
    rl.close();
  }
};

// Only run if this file is executed directly
main().catch((error) => {
  console.error(chalk.red("Unhandled error:"), error);
  process.exit(1);
});

// Some Prepared questions:

// Hello, I'm Oliver. I'm anxious about starting to learn in Scrimba and have a couple of questions.
// First of all, what can I learn in Scrimba?
// Do you have some community where I can go and ask questions?
