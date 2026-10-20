"""Takes the demo's project off its billing account, which stops everything in it that costs money.

Two messages on the same Pub/Sub topic trigger it: the budget's notifications, which say what has been spent
and what the budget is, and one from Cloud Scheduler on the demo's last day, which says {"deadline": true}.
Anything else is ignored. The data stays; linking the billing account again brings the project back.
"""

import base64
import json
import os

import functions_framework

PROJECT = os.environ["PROJECT_ID"]


def should_stop(message: dict) -> bool:
    if message.get("deadline"):
        return True
    spent, budget = message.get("costAmount"), message.get("budgetAmount")
    return spent is not None and budget is not None and spent >= budget


@functions_framework.cloud_event
def stop(event) -> None:
    message = json.loads(base64.b64decode(event.data["message"]["data"]))
    if not should_stop(message):
        print(f"kept running: {message}")
        return
    from google.cloud import billing_v1

    # No check of the current state first: reading it needs resourcemanager.projects.get, which the function's
    # role (Project Billing Manager) does not have. Unlinking an unlinked project changes nothing.
    billing_v1.CloudBillingClient().update_project_billing_info(
        name=f"projects/{PROJECT}", project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name=""))
    print(f"billing turned off for {PROJECT}: {message}")


if __name__ == "__main__":
    # python main.py   checks the decision; PROJECT_ID can be anything here.
    assert should_stop({"deadline": True})
    assert should_stop({"costAmount": 400000, "budgetAmount": 400000})
    assert not should_stop({"costAmount": 399999.5, "budgetAmount": 400000})
    assert not should_stop({"budgetDisplayName": "secop-analyst demo"})  # a message without amounts
    print("ok")
