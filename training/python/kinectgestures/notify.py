import requests

"""
Support for training notifications via Slack. 
To set this up, set the WEBHOOK_URL to a url you can create in your Slack workspace. 
"""

WEBHOOK_URL = None


def webhook_not_set():
    print("Skipped sending Slack notification. To setup Slack notifications, edit notify.py (it's easy!)")
    return False


def notify_start_trainings(configs):
    if not WEBHOOK_URL:
        return webhook_not_set()

    requests.post(WEBHOOK_URL, json={
        "text": "Started running {} experiments".format(len(configs))
    })
    print("Sent notification about training start")


def notify_all_trainings_done(configs):
    if not WEBHOOK_URL:
        return webhook_not_set()

    payload = {
        "text": "Finished {} experiments".format(len(configs))
    }
    requests.post(WEBHOOK_URL, json=payload)
    print("Sent notification about finished training.")


def notify_training_crashed(exception):
    if not WEBHOOK_URL:
        return webhook_not_set()

    requests.post(WEBHOOK_URL, json={
        "text": "Oops. Training has crashed. Message: {}".format(exception)
    })
    print("Sent crash notification")


if __name__ == '__main__':
    try:
        l = [1]
        print(l[3])
    except Exception as e:
        notify_training_crashed(e)
        raise e
