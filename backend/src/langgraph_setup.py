from typing import Any, Dict, TypedDict

from langchain_core.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

from .llm_handler import LLMHandler
from .logger import logger


# Define the state schema
class State(TypedDict):
    """State definition for the graph."""

    query: str
    context: str
    initial_answer: str | None
    reasoning: str | None
    final_answer: Dict[str, Any] | None


class LangGraphSetup:
    """Setup class for creating and managing the LangGraph workflow."""

    def __init__(self):
        """Initialize the LangGraph setup with necessary components."""
        self.llm_handler = LLMHandler()

        # Define prompt templates
        self.qa_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Answer the following question based on the provided context. 
            If you cannot find the answer in the context, say "I don't have enough information to answer this question."
            
            Context: {context}
            Question: {query}
            
            Answer:""",
        )

        self.reasoning_template = PromptTemplate(
            input_variables=["query", "context", "initial_answer"],
            template="""
            Given the following question and context, provide a step-by-step reasoning process:
            
            Question: {query}
            Context: {context}
            Initial Answer: {initial_answer}
            
            Reasoning steps:""",
        )

    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow graph.

        Returns:
            StateGraph: Compiled workflow graph
        """
        # Initialize the state graph
        workflow = StateGraph(State)

        # Create LLM chains
        qa_chain = LLMChain(llm=self.llm_handler.llm, prompt=self.qa_template)
        reasoning_chain = LLMChain(llm=self.llm_handler.llm, prompt=self.reasoning_template)

        # Define node functions
        def generate_initial_answer(state: State) -> State:
            """Generate initial answer using the QA chain."""
            try:
                response = qa_chain.run(query=state["query"], context=state["context"])
                state["initial_answer"] = response
                return state
            except Exception as e:
                logger.error(f"Error in generate_initial_answer: {str(e)}")
                state["initial_answer"] = "Error generating answer"
                return state

        def generate_reasoning(state: State) -> State:
            """Generate reasoning based on the initial answer."""
            try:
                response = reasoning_chain.run(
                    query=state["query"],
                    context=state["context"],
                    initial_answer=state["initial_answer"],
                )
                state["reasoning"] = response
                return state
            except Exception as e:
                logger.error(f"Error in generate_reasoning: {str(e)}")
                state["reasoning"] = "Error generating reasoning"
                return state

        def create_final_response(state: State) -> State:
            """Combine outputs and calculate confidence."""
            try:
                initial_answer = state.get("initial_answer", "")
                reasoning = state.get("reasoning", "")

                confidence = self._calculate_confidence(initial_answer, reasoning)

                state["final_answer"] = {
                    "answer": initial_answer,
                    "reasoning": reasoning,
                    "confidence": confidence,
                }
                return state
            except Exception as e:
                logger.error(f"Error in create_final_response: {str(e)}")
                state["final_answer"] = {
                    "answer": "Error processing response",
                    "reasoning": "",
                    "confidence": 0.0,
                }
                return state

        # Add nodes to the workflow
        workflow.add_node("generate_answer", generate_initial_answer)
        workflow.add_node("generate_reasoning", generate_reasoning)
        workflow.add_node("create_final_response", create_final_response)

        # Define the workflow structure
        workflow.set_entry_point("generate_answer")
        workflow.add_edge("generate_answer", "generate_reasoning")
        workflow.add_edge("generate_reasoning", "create_final_response")
        workflow.set_finish_point("create_final_response")

        # Compile and return the workflow
        return workflow.compile()

    def _calculate_confidence(self, answer: str, reasoning: str) -> float:
        """
        Calculate confidence score for the response.

        Args:
            answer: Initial answer from QA chain
            reasoning: Reasoning steps from reasoning chain

        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence = 1.0

            # Reduce confidence for uncertainty indicators
            uncertainty_phrases = [
                "I don't have enough information",
                "I'm not sure",
                "unclear",
                "cannot determine",
                "possibly",
                "might be",
            ]

            for phrase in uncertainty_phrases:
                if phrase.lower() in answer.lower():
                    confidence *= 0.5

            # Adjust confidence based on reasoning quality
            if reasoning:
                words = len(reasoning.split())
                if words > 50:  # Detailed reasoning
                    confidence *= 1.2
                elif words < 20:  # Brief reasoning
                    confidence *= 0.8

            # Ensure confidence stays within bounds
            return min(max(confidence, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5


def create_graph() -> StateGraph:
    """
    Factory function to create and return a configured LangGraph instance.

    Returns:
        StateGraph: Configured processing graph
    """
    setup = LangGraphSetup()
    return setup.create_workflow()
