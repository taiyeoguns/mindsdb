import shopify
import requests
import pandas as pd
from typing import Text, List, Dict, Any

from mindsdb_sql.parser import ast
from mindsdb.integrations.libs.api_handler import APITable

from mindsdb.integrations.handlers.utilities.query_utilities import SELECTQueryParser, SELECTQueryExecutor
from mindsdb.integrations.handlers.utilities.query_utilities import INSERTQueryParser
from mindsdb.integrations.handlers.utilities.query_utilities import DELETEQueryParser, DELETEQueryExecutor
from mindsdb.integrations.handlers.utilities.query_utilities import UPDATEQueryParser, UPDATEQueryExecutor

from mindsdb.utilities.log import get_log

logger = get_log("integrations.shopify_handler")


class ProductsTable(APITable):
    """The Shopify Products Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /products" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame
            Shopify Products matching the query

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """

        select_statement_parser = SELECTQueryParser(
            query,
            'products',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        products_df = pd.json_normalize(self.get_products(limit=result_limit))

        select_statement_executor = SELECTQueryExecutor(
            products_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )
        products_df = select_statement_executor.execute_query()

        return products_df
    
    def insert(self, query: ast.Insert) -> None:
        """Inserts data into the Shopify "POST /products" API endpoint.

        Parameters
        ----------
        query : ast.Insert
           Given SQL INSERT query

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        insert_statement_parser = INSERTQueryParser(
            query,
            supported_columns=['title', 'body_html', 'vendor', 'product_type', 'tags', 'status'],
            mandatory_columns=['title' ],
            all_mandatory=False
        )
        product_data = insert_statement_parser.parse_query()
        self.create_products(product_data)
    
    def delete(self, query: ast.Delete) -> None:
        """
        Deletes data from the Shopify "DELETE /products" API endpoint.

        Parameters
        ----------
        query : ast.Delete
           Given SQL DELETE query

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        delete_statement_parser = DELETEQueryParser(query)
        where_conditions = delete_statement_parser.parse_query()

        products_df = pd.json_normalize(self.get_products())

        delete_query_executor = DELETEQueryExecutor(
            products_df,
            where_conditions
        )

        products_df = delete_query_executor.execute_query()

        product_ids = products_df['id'].tolist()
        self.delete_products(product_ids)

    def get_columns(self) -> List[Text]:
        return pd.json_normalize(self.get_products(limit=1)).columns.tolist()

    def get_products(self, **kwargs) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        products = shopify.Product.find(**kwargs)
        return [product.to_dict() for product in products]
    
    def delete_products(self, product_ids: List[int]) -> None:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)

        for product_id in product_ids:
            product = shopify.Product.find(product_id)
            product.destroy()
            logger.info(f'Product {product_id} deleted')


    def create_products(self, product_data: List[Dict[Text, Any]]) -> None:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)

        for product in product_data:
            created_product = shopify.Product.create(product)
            if 'id' not in created_product.to_dict():
                raise Exception('Product creation failed')
            else:
                logger.info(f'Product {created_product.to_dict()["id"]} created')


class CustomersTable(APITable):
    """The Shopify Customers Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /customers" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame
            Shopify Customers matching the query

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'customers',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        customers_df = pd.json_normalize(self.get_customers(limit=result_limit))

        select_statement_executor = SELECTQueryExecutor(
            customers_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )
        customers_df = select_statement_executor.execute_query()

        return customers_df

    def insert(self, query: ast.Insert) -> None:
        """Inserts data into the Shopify "POST /customers" API endpoint.

        Parameters
        ----------
        query : ast.Insert
           Given SQL INSERT query

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        insert_statement_parser = INSERTQueryParser(
            query,
            supported_columns=['first_name', 'last_name', 'email', 'phone', 'tags', 'currency'],
            mandatory_columns=['first_name', 'last_name', 'email', 'phone'],
            all_mandatory=False
        )
        customer_data = insert_statement_parser.parse_query()
        self.create_customers(customer_data)

    def update(self, query: ast.Update) -> None:
        """Updates data in the Shopify "PUT /customers" API endpoint.

        Parameters
        ----------
        query : ast.Update
           Given SQL UPDATE query

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        update_statement_parser = UPDATEQueryParser(query)
        values_to_update, where_conditions = update_statement_parser.parse_query()

        customers_df = pd.json_normalize(self.get_customers())

        update_query_executor = UPDATEQueryExecutor(
            customers_df,
            where_conditions
        )

        customers_df = update_query_executor.execute_query()

        customer_ids = customers_df['id'].tolist()

        self.update_customers(customer_ids, values_to_update)

    def get_columns(self) -> List[Text]:
        return pd.json_normalize(self.get_customers(limit=1)).columns.tolist()

    def get_customers(self, **kwargs) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        customers = shopify.Customer.find(**kwargs)
        return [customer.to_dict() for customer in customers]

    def create_customers(self, customer_data: List[Dict[Text, Any]]) -> None:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)

        for customer in customer_data:
            created_customer = shopify.Customer.create(customer)
            if 'id' not in created_customer.to_dict():
                raise Exception('Customer creation failed')
            else:
                logger.info(f'Customer {created_customer.to_dict()["id"]} created')

    def update_customers(self, customer_ids: List[int], values_to_update: List[Dict[Text, Any]]) -> None:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)

        for customer_id in customer_ids:
            customer = shopify.Customer.find(customer_id)
            for key, value in values_to_update.items():
                setattr(customer, key, value)
            customer.save()
            logger.info(f'Customer {customer_id} updated')


class OrdersTable(APITable):
    """The Shopify Orders Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /orders" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame
            Shopify Orders matching the query

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'orders',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()
        orders_df = pd.json_normalize(self.get_orders(limit=result_limit))

        select_statement_executor = SELECTQueryExecutor(
            orders_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )
        orders_df = select_statement_executor.execute_query()

        return orders_df

    def get_columns(self) -> List[Text]:
        return pd.json_normalize(self.get_orders(limit=1)).columns.tolist()

    def get_orders(self, **kwargs) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        orders = shopify.Order.find(**kwargs)
        return [order.to_dict() for order in orders]

class InventoryLevelTable(APITable):
    """The Shopify Inventory Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /inventory" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame

            Shopify Inventory matching the inventory_item_id or/and location_id

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'inventory_level',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        search_params = {}
        subset_where_conditions = []
        for op, arg1, arg2 in where_conditions:
            if arg1 == 'inventory_item_ids':
                if op == '=':
                    search_params["inventory_item_ids"] = arg2
                else:
                    raise NotImplementedError("Only '=' operator is supported for inventory_item_ids column.")
            elif arg1 == 'location_ids':
                if op == '=':
                    search_params["location_ids"] = arg2
                else:
                    raise NotImplementedError("Only '=' operator is supported for location_ids column.")
            elif arg1 in ['available', 'updated_at']:
                subset_where_conditions.append([op, arg1, arg2])

        filter_flag = ("inventory_item_ids" in search_params) or ("location_ids" in search_params)

        if not filter_flag:
            raise NotImplementedError("inventory_item_ids or location_ids column has to be present in where clause.")

        search_params["limit"] = result_limit

        inventory_df = pd.json_normalize(self.get_inventory(search_params))

        self.clean_selected_columns(selected_columns)

        select_statement_executor = SELECTQueryExecutor(
            inventory_df,
            selected_columns,
            subset_where_conditions,
            order_by_conditions
        )
        inventory_df = select_statement_executor.execute_query()

        return inventory_df

    def clean_selected_columns(self, selected_cols) -> List[Text]:
        if "inventory_item_ids" in selected_cols:
            selected_cols.remove("inventory_item_ids")
            selected_cols.append("inventory_item_id")
        if "location_ids" in selected_cols:
            selected_cols.remove("location_ids")
            selected_cols.append("location_id")

    def get_columns(self) -> List[Text]:
        return ["inventory_item_ids", "location_ids", "available", "updated_at"]

    def get_inventory(self, kwargs) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        inventories = shopify.InventoryLevel.find(**kwargs)
        return [inventory.to_dict() for inventory in inventories]

class LocationTable(APITable):
    """The Shopify Location Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /locations" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame

            Shopify Locations matching the query

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'locations',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        locations_df = pd.json_normalize(self.get_locations(limit=result_limit))

        select_statement_executor = SELECTQueryExecutor(
            locations_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )

        locations_df = select_statement_executor.execute_query()

        return locations_df

    def get_columns(self) -> List[Text]:
        return pd.json_normalize(self.get_locations(limit=1)).columns.tolist()

    def get_locations(self, **kwargs) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        locations = shopify.Location.find(**kwargs)
        return [location.to_dict() for location in locations]

class CustomerReviews(APITable):
    """The Shopify Customer Reviews Table implementation"""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Yotpo "GET https://api.yotpo.com/v1/apps/{app_key}/reviews?utoken={utoken}" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame

            Shopify customer reviews matching the query

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'customer_reviews',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        customer_reviews_df = pd.json_normalize(self.get_customer_reviews(limit=result_limit))

        select_statement_executor = SELECTQueryExecutor(
            customer_reviews_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )

        customer_reviews_df = select_statement_executor.execute_query()

        return customer_reviews_df

    def get_columns(self) -> List[Text]:
        return pd.json_normalize(self.get_customer_reviews(limit=1)).columns.tolist()

    def get_customer_reviews(self, **kwargs) -> List[Dict]:
        if self.handler.yotpo_app_key is None or self.handler.yotpo_access_token is None:
            raise Exception("You need to provide 'yotpo_app_key' and 'yotpo_access_token' to retrieve customer reviews.")
        url = f"https://api.yotpo.com/v1/apps/{self.handler.yotpo_app_key}/reviews?count=0&utoken={self.handler.yotpo_access_token}"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        json_response = requests.get(url, headers=headers).json()
        return [review for review in json_response['reviews']] if 'reviews' in json_response else []

class CarrierServiceTable(APITable):
    """The Shopify carrier service Table implementation. Example carrier services like usps, dhl etc."""

    def select(self, query: ast.Select) -> pd.DataFrame:
        """Pulls data from the Shopify "GET /carrier_services" API endpoint.

        Parameters
        ----------
        query : ast.Select
           Given SQL SELECT query

        Returns
        -------
        pd.DataFrame

            Shopify Carrier service info

        Raises
        ------
        ValueError
            If the query contains an unsupported condition
        """
        select_statement_parser = SELECTQueryParser(
            query,
            'carrier_service',
            self.get_columns()
        )
        selected_columns, where_conditions, order_by_conditions, result_limit = select_statement_parser.parse_query()

        carrier_service_df = pd.json_normalize(self.get_carrier_service())

        select_statement_executor = SELECTQueryExecutor(
            carrier_service_df,
            selected_columns,
            where_conditions,
            order_by_conditions
        )
        
        carrier_service_df = select_statement_executor.execute_query()

        return carrier_service_df

    def get_columns(self) -> List[Text]:
        return ["id", "name", "active", "service_discovery", "carrier_service_type", "admin_graphql_api_id"]

    def get_carrier_service(self) -> List[Dict]:
        api_session = self.handler.connect()
        shopify.ShopifyResource.activate_session(api_session)
        services = shopify.CarrierService.find()
        return [service.to_dict() for service in services]