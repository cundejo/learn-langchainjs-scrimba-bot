import "dotenv/config";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { models } from "./models.js";

/**
 * This is an example on how to use Runnable.
 *
 * In this small example we will receive a text, and:
 * 1. Fix the grammar errors in the text
 * 2. And translate it to another language
 *
 * Docs:
 *  - https://js.langchain.com/docs/how_to/lcel_cheatsheet/
 *  - https://js.langchain.com/docs/concepts/runnables
 */

const llm = models.openai();

const grammarTemplate = "Fix the grammar errors in the following text: {text}";
const grammarPrompt = PromptTemplate.fromTemplate(grammarTemplate);

const translationTemplate =
  "Translate the following text to {language}: {text}";
const translationPrompt = PromptTemplate.fromTemplate(translationTemplate);

const grammarChain = RunnableSequence.from([
  grammarPrompt,
  // logger,
  llm,
  new StringOutputParser(),
]);

const translationChain = RunnableSequence.from([
  translationPrompt,
  llm,
  new StringOutputParser(),
]);

const chain = RunnableSequence.from([
  {
    text: grammarChain,
    language: (input) => input.language,
  },
  // logger,
  translationChain,
]);

const response = await chain.invoke({
  text: "i dont liked mondays",
  language: "Spanish",
});

console.log(response);
