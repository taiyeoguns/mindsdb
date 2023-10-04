import pandas as pd
from mindsdb.integrations.libs.api_handler import APITable
from mindsdb_sql.parser import ast
from mindsdb.integrations.utilities.sql_utils import extract_comparison_conditions
from typing import List, Tuple
import requests


class StoriesTable(APITable):
    def select(self, query: ast.Select) -> pd.DataFrame:
        """Select data from the stories table and return it as a pandas DataFrame.
        Args:
            query (ast.Select): The SQL query to be executed.
        Returns:
            pandas.DataFrame: A pandas DataFrame containing the selected data.
        """
        hn_handler = self.handler

        limit = query.limit.value if query.limit is not None else None
        # Call the Hacker News API to get the top stories
        url = f'{hn_handler.base_url}/topstories.json'
        response = requests.get(url)
        data = response.json()

        # Fetch the details of the top stories, up to the specified limit
        stories_data = []
        for story_id in data[:limit]:
            url = f'{hn_handler.base_url}/item/{story_id}.json'
            response = requests.get(url)
            story_data = response.json()
            stories_data.append(story_data)

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(
            stories_data, 
            columns=['id', 'time', 'title', 'url', 'score', 'descendants']
            )

        # Apply any WHERE clauses in the SQL query to the DataFrame
        conditions = extract_comparison_conditions(query.where)
        for condition in conditions:
            if condition[0] == '=' and condition[1] == 'id':
                df = df[df['id'] == int(condition[2])]
            elif condition[0] == '>' and condition[1] == 'time':
                timestamp = int(condition[2])
                df = df[df['time'] > timestamp]

        # Filter the columns in the DataFrame according to the SQL query
        self.filter_columns(df, query)

        return df


    def get_columns(self):
        """Get the list of column names for the stories table.
        Returns:
            list: A list of column names for the stories table.
        """
        return ['id', 'time', 'title', 'url', 'score', 'descendants']

    def filter_columns(self, df, query):
        """Filter the columns in the DataFrame according to the SQL query.
        Args:
            df (pandas.DataFrame): The DataFrame to filter.
            query (ast.Select): The SQL query to apply to the DataFrame.
        """
        columns = []
        for target in query.targets:
            if isinstance(target, ast.Star):
                columns = self.get_columns()
                break
            elif isinstance(target, ast.Identifier):
                columns.append(target.value)
        df = df[columns]
        return df


class CommentsTable(APITable):
    def select(self, query: ast.Select) -> pd.DataFrame:
        """Select data from the comments table and return it as a pandas DataFrame.
        Args:
            query (ast.Select): The SQL query to be executed.
        Returns:
            pandas.DataFrame: A pandas DataFrame containing the selected data.
        """
        hn_handler = self.handler

        limit = query.limit.value if query.limit is not None else None
        # Get the item ID from the SQL query
        item_id = None
        conditions = extract_comparison_conditions(query.where)
        for condition in conditions:
            if condition[0] == '=' and condition[1] == 'item_id':
                item_id = condition[2]

        if item_id is None:
            raise ValueError('Item ID is missing in the SQL query')

        # Call the Hacker News API to get the comments for the specified item
        comments_df = hn_handler.call_hackernews_api('get_comments', params={'item_id': item_id})

        # Fill NaN values with 'deleted'
        comments_df = comments_df.fillna('deleted')
        # Filter the columns to those specified in the SQL query
        self.filter_columns(comments_df, query)

        # Limit the number of results if necessary
        if limit is not None:
            comments_df = comments_df.head(limit)

        return comments_df
    
    
    def get_columns(self) -> List[str]:
        """Get the list of column names for the comments table.
        Returns:
            list: A list of column names for the comments table.
        """
        return [
            'id',
            'by',
            'parent',
            'text',
            'time',
            'type',
        ]

    def filter_columns(self, result: pd.DataFrame, query: ast.Select = None) -> None:
        """Filter the columns of a DataFrame to those specified in an SQL query.
        Args:
            result (pandas.DataFrame): The DataFrame to filter.
            query (ast.Select): The SQL query containing the column names to filter on.
        """
        if query is None:
            return

        columns = []
        for target in query.targets:
            if isinstance(target, ast.Star):
                return
            elif isinstance(target, ast.Identifier):
                columns.append(target.value)

        if columns:
            result = result[columns]
