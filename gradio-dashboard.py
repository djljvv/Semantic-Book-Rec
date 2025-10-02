import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings()) #converts to document embeddings using the open ai embeddings we made and stores in chroma data base

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50, #initially retrieve 50 recs, apply filtering, and then have final top 16
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k) #do similarity search, limit to top k
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs] #get isbns and split them off the descriptions
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k) #limit books data frame to just those that match isbns

    #filtering(drop down for dashboard)
    if category != "ALL":#(anything other than all, filter down to specific categories
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    #sort based on probabilities, just a bunch of if statments that sort by the specific emotions
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    recomendations = retrieve_semantic_recommendations(query, category, tone)#get data frame
    results = [] #empty list

    for _, row in recomendations.iterrows():#loop over all recommendations
        description = row["description_x"]
        truncated_desc_split = description.split() #split it up into seperate words
        truncated_description = " ". join(truncated_desc_split[:30]) + " ..." #if it has more than 30 words, cur off and trail with elipses

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"#basically adds all authors up until the last one and then adds them with "and this guy"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}" #how we display the information thats appended to the bottom of the book image
        results.append((row["large_thumbnail"], caption)) #tuple containing thumbnail and caption

    return results

categories = ["ALL"] + sorted(books["simple_categories"].unique())
tones = ["ALL"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("#Semantic Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "PLease enter a description of a desired book:",
                                placeholder = "e.g., A story about love")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category: ", value = "ALL")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select a tone: ", value = "ALL")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended Books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)

if __name__ == "__main__":
    dashboard.launch()