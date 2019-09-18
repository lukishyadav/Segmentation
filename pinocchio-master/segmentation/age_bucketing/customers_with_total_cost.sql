SELECT
"object_data-customer_id",
sum("object_data-total_to_charge")

FROM "data_lake_us_prod"."sa_object_changed"
WHERE "name" = 'PAYMENT_LIFECYCLE'
and tenant_id = 'darwin-prod'
and "object_data-total_to_charge" > 0
group by "object_data-customer_id"