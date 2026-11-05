import os
import stat
import threading

import pytest

from src.data.accounts import AccountError, AccountStore, hash_password, verify_password


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "test.db")


def make_thread(store, user, thread_id="t1", title="First question"):
    store.ensure_thread(user["id"], thread_id, title)
    return thread_id


def test_password_is_salted_and_verifies():
    first, second = hash_password("correct horse"), hash_password("correct horse")
    assert first != second
    assert "correct horse" not in first
    assert verify_password("correct horse", first)
    assert not verify_password("wrong", first)


def test_database_never_contains_the_password(store):
    store.create_user("a@example.com", "plaintext-secret")
    assert b"plaintext-secret" not in store.path.read_bytes()


def test_database_is_private_to_its_owner(store):
    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600


def test_login_normalizes_email(store):
    created = store.create_user(" A@Example.com ", "password1")
    assert store.authenticate("a@example.COM", "password1") == created


@pytest.mark.parametrize("email,password", [("a@example.com", "wrong-pass"), ("nobody@example.com", "password1")])
def test_failed_login_looks_the_same(store, email, password):
    store.create_user("a@example.com", "password1")
    assert store.authenticate(email, password) is None


def test_overlong_password_is_refused(store):
    store.create_user("a@example.com", "password1")
    assert store.authenticate("a@example.com", "x" * 2000) is None


@pytest.mark.parametrize("email,password", [
    ("not-an-email", "password1"), ("a@b", "password1"), ("a b@example.com", "password1"),
    ("a@example.com", "short"), ("a@example.com", "x" * 2000),
])
def test_bad_signup_is_refused(store, email, password):
    with pytest.raises(AccountError):
        store.create_user(email, password)


def test_duplicate_email_is_refused(store):
    store.create_user("a@example.com", "password1")
    with pytest.raises(AccountError):
        store.create_user("A@EXAMPLE.COM", "password2")


def test_profile_round_trip_is_per_user(store):
    a, b = store.create_user("a@example.com", "password1"), store.create_user("b@example.com", "password1")
    store.save_profile(a["id"], {"risk_tolerance": "aggressive"})
    assert store.get_profile(a["id"]) == {"risk_tolerance": "aggressive"}
    assert store.get_profile(b["id"]) == {}


def test_turns_keep_their_details_and_survive_reopening(store):
    a = store.create_user("a@example.com", "password1")
    thread = make_thread(store, a)
    store.add_turn(a["id"], thread, "user", "What is an ETF?")
    store.add_turn(a["id"], thread, "assistant", "A fund.", sources=["SEC"], caption="c", agents=["finance_qa"])
    turns = AccountStore(store.path).get_turns(a["id"], thread)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[1]["sources"] == ["SEC"] and turns[1]["agents"] == ["finance_qa"] and turns[1]["rating"] is None


def test_threads_list_newest_first_with_short_titles(store):
    a = store.create_user("a@example.com", "password1")
    make_thread(store, a, "old", "x" * 200)
    make_thread(store, a, "new", "second")
    make_thread(store, a, "old", "ignored on later turns")
    threads = store.list_threads(a["id"])
    assert threads[0]["id"] == "old" and threads[0]["title"] == "x" * 60


def test_other_users_cannot_touch_a_thread(store):
    a, b = store.create_user("a@example.com", "password1"), store.create_user("b@example.com", "password1")
    thread = make_thread(store, a)
    store.add_turn(a["id"], thread, "assistant", "private", agents=["tax"])
    turn_id = store.get_turns(a["id"], thread)[0]["id"]

    assert not store.owns_thread(b["id"], thread)
    assert store.list_threads(b["id"]) == []
    assert store.get_turns(b["id"], thread) == []
    assert not store.rate_turn(b["id"], turn_id, 1)
    assert not store.delete_thread(b["id"], thread)
    with pytest.raises(PermissionError):
        store.ensure_thread(b["id"], thread, "take over")
    with pytest.raises(PermissionError):
        store.add_turn(b["id"], thread, "user", "injected")

    turns = store.get_turns(a["id"], thread)
    assert len(turns) == 1 and turns[0]["rating"] is None
    assert store.list_threads(a["id"])[0]["title"] == "First question"


def test_only_the_first_rating_counts(store):
    a = store.create_user("a@example.com", "password1")
    thread = make_thread(store, a)
    store.add_turn(a["id"], thread, "assistant", "answer", agents=["tax"])
    turn_id = store.get_turns(a["id"], thread)[0]["id"]
    assert store.rate_turn(a["id"], turn_id, 1)
    assert not store.rate_turn(a["id"], turn_id, 0)
    assert store.get_turns(a["id"], thread)[0]["rating"] == 1


def test_deleting_a_thread_removes_its_turns(store):
    a = store.create_user("a@example.com", "password1")
    thread = make_thread(store, a)
    store.add_turn(a["id"], thread, "user", "question")
    assert store.delete_thread(a["id"], thread)
    assert store.list_threads(a["id"]) == [] and store.get_turns(a["id"], thread) == []
    make_thread(store, a)  # the ID is free again, with nothing left over
    assert store.get_turns(a["id"], thread) == []


def test_parallel_signups_for_one_email_create_one_account(store):
    results = []

    def sign_up():
        try:
            results.append(store.create_user("a@example.com", "password1"))
        except AccountError:
            pass

    threads = [threading.Thread(target=sign_up) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 1
