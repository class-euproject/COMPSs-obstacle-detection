def main():
    from dataclay.tool.functions import get_stubs
    import subprocess

    # Env variables
    dataclay_jar_path = "dataclay/dataclay.jar"
    user = "CityUser"
    password = "p4ssw0rd"
    namespace = "CityNS"
    stubspath = "./stubs"
    
    # Connection to get contract_id
    contract_id = subprocess.check_output(f"java -cp {dataclay_jar_path} es.bsc.dataclay.tool.AccessNamespace {user} {password} {namespace} | tail -1", shell=True)[:-1].decode()

    # Get stubs
    get_stubs(user, password, contract_id, stubspath)


if __name__ == "__main__":
    main()
