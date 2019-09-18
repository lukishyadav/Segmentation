SELECT distinct "tenant_id",
"object_data-rental_id",
from_iso8601_timestamp("object_data-rental_started_at") as "Rental_start_time",
from_iso8601_timestamp("object_data-rental_booked_at") as "Booking Start Time",
from_iso8601_timestamp("object_data-rental_ended_at") as "Rental_end_time"
FROM "data_lake_us_prod"."sa_object_changed"
WHERE "name" = 'RENTAL_LIFECYCLE'
and tenant_id = 'darwin-prod'
and "object_data-status" = 'ENDED'
AND "object_data-reservation_cancelled_at" is null
and "object_data-rental_started_at" > '2019-02-01 00 (tel:2019020100):00:01'