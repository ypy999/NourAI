from data.readiness import CONDITIONALLY_READY, NOT_READY, READY, assess_corpus


def test_readiness_separates_integrity_license_and_language_coverage():
    checks={"valid":True}
    conditional=assess_corpus(technical_checks=checks,license_statuses=["VERIFIED"],language_coverage={"Arabic":0,"English":100})
    assert conditional["status"]==CONDITIONALLY_READY and not conditional["training_ready"]
    ready=assess_corpus(technical_checks=checks,license_statuses=["VERIFIED"],language_coverage={"Arabic":1,"English":100})
    assert ready["status"]==READY
    blocked=assess_corpus(technical_checks={"valid":False},license_statuses=["UNKNOWN_LICENSE"])
    assert blocked["status"]==NOT_READY and not blocked["size_threshold_used"]
