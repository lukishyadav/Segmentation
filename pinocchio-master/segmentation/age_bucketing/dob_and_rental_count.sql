select distinct "object_data-customer_id",
"object_data-date_of_birth",
max("object_data-num_of_paid_drives")
from data_lake_us_prod."sa_object_changed"
where tenant_id='darwin-prod'
and "object_data-exempt_from_payment" = False
and "object_data-is_staff" = False
and "object_data-date_of_birth" is not null
and "object_data-can_reserve_car" = True
and "object_data-num_of_paid_drives" >= 1
and "object_data-email" not like '%aaa%'
and "object_data-email" not like '%ridecell%'
and "object_data-email" not like '%gig%'
and "object_data-email" not like '%eco%'
group by "object_data-customer_id", "object_data-date_of_birth"