def main():
    from dataclay.tool.functions import get_stubs
    import subprocess
    import time

    # Env variables
    dataclay_jar_path = "dataclay/dataclay.jar"
    user = "CityUser"
    password = "p4ssw0rd"
    namespace = "CityNS"
    stubspath = "./stubs"
    
    # Connection to get contract_id
    contract_id = None
    while contract_id is None:
        try:
            contract_id = subprocess.check_output(f"java -cp {dataclay_jar_path} es.bsc.dataclay.tool.AccessNamespace {user} {password} {namespace} | tail -1", shell=True)[:-1].decode()
            print(f"CONTRACT ID IS {contract_id}")
            if contract_id == "":
                contract_id = None
        except:
            contract_id = None
            print(f"Waiting for contract_id to be set and ready from dataclay...")
            time.sleep(1)

    # Get stubs
    print(f"CONTRACT ID OUTSIDE WHILE IS {contract_id}")
    get_stubs(user, password, contract_id, stubspath)


if __name__ == "__main__":
    main()
