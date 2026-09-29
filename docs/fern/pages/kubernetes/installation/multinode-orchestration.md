---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multinode Orchestration
---

Multinode deployments require either Grove + KAI Scheduler or an alternative orchestrator setup (LeaderWorkerSet + Volcano) to enable gang scheduling for workloads that span multiple nodes.

## Option 1: Grove + KAI Scheduler

Grove is the default and recommended orchestrator for multinode deployments. It requires KAI Scheduler as well. There are two ways to enable Grove and KAI Scheduler, either dynamo can install it automatically (recommended for development and testing), or you can install them separately (recommended for production).

<Tabs>
  <Tab title="Managed Installation" value="managed">

  The managed installation is recommended for development and testing. It is the simplest path, and allows Dynamo to manage the lifecycle of Grove and KAI Scheduler as bundled subcharts. Run the following command to install Dynamo with Grove and KAI Scheduler:

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.install=true" \
  --set "global.kai-scheduler.install=true" \
  --set-string "kai-scheduler.scheduler.args.default-staleness-grace-period=-1s"
  ```
  </Tab>
  <Tab title="External Installation" value="external">
  The external installation is recommended for production or if it's already installed on your system. It allows you to install Grove and KAI Scheduler separately, and manage their lifecycle independently or share them across namespaces.

  See the [Grove installation guide](https://github.com/NVIDIA/grove/blob/main/docs/installation.md) and [KAI Scheduler deployment guide](https://github.com/NVIDIA/KAI-Scheduler) for instructions.

  Then, run the following command to install or configure Dynamo to use the existing Grove and KAI Scheduler:

  ```bash
  helm upgrade --install dynamo-platform dynamo-platform-$RELEASE_VERSION.tgz \
  --namespace $NAMESPACE \
  --create-namespace \
  --set "global.grove.enabled=true" \
  --set "global.kai-scheduler.enabled=true"
  ```

  > [!NOTE]
  > If you install Grove and KAI Scheduler externally, ensure that the versions are compatible with the Dynamo Platform version you are installing. The following table shows the minimum required versions for each component:
  >
  > | dynamo-platform | kai-scheduler | Grove |
  > |-----------------|---------------|-------|
  > | 1.0.x           | >= v0.13.0    | >= v0.1.0-alpha.6 |
  > | 1.1.x           | >= v0.13.4    | >= v0.1.0-alpha.8 |
  > | 1.5.x           | >= v0.17.0    | >= v0.1.0-alpha.13 |
  > | 1.6.x           | >= v0.17.0    | >= v0.1.0-alpha.14-rc1 |

  </Tab>
</Tabs>


## Option 2: LeaderWorkerSet + Volcano

If you are not using Grove for multinode, you can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/installation/) (>= v0.7.0) with [Volcano](https://github.com/volcano-sh/volcano#quick-start-guide) for gang scheduling. Both must be installed before deploying multinode workloads.

1. Install Volcano:

```bash
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

2. Install LWS (>= v0.7.0) with Volcano gang scheduling enabled:

```bash
export LWS_VERSION=0.8.0
helm install lws oci://registry.k8s.io/lws/charts/lws \
  --version=$LWS_VERSION \
  --namespace lws-system \
  --create-namespace \
  --set gangSchedulingManagement.schedulerProvider=volcano \
  --wait --timeout 300s
```

See the [LWS docs](https://lws.sigs.k8s.io/docs/) and [Volcano docs](https://github.com/volcano-sh/volcano#quick-start-guide) for configuration options.

## Portable Topology Environment Variables

**Available since Dynamo 1.6.0.** The operator injects two provider-independent environment
variables into the `main` container of each new multinode DynamoGraphDeployment (DGD) component:

| Variable | Meaning | Grove source | LWS source |
|---|---|---|---|
| `DYNAMO_RANK` | Zero-based engine node rank. The leader is `0`; workers use `1` through `multinode.nodeCount - 1`. | `GROVE_PCSG_POD_INDEX` | `LWS_WORKER_INDEX` |
| `DYNAMO_LEADER_ADDRESS` | DNS hostname of the engine leader, without a port. | PodCliqueScalingGroup leader DNS name | `LWS_LEADER_ADDRESS` |

For an inter-pod GPU Memory Service (GMS) deployment, the operator sets `DYNAMO_RANK` from the
engine role instead of the PCSG-wide pod index because the scaling group also contains weight-server
cliques.

The operator owns these variables and replaces values with the same names from the component's
`podTemplate`. Use the aliases instead of provider-specific `GROVE_*` or `LWS_*` variables when a
custom command needs the component topology.

Kubernetes expands `$(NAME)` references in a container's `command` and `args`. A shell process that
runs inside the container reads the same environment variables with `$NAME` or `${NAME}`. For
example, an engine command can receive the topology as separate arguments:

```yaml
args:
- --master-addr
- $(DYNAMO_LEADER_ADDRESS)
- --master-port
- "29500"
- --node-rank
- $(DYNAMO_RANK)
```

> [!NOTE]
> Do not add these flags to standard automatically configured vLLM or SGLang components. In vLLM
> multiprocessing mode, the operator uses the aliases for `--master-addr` and `--node-rank`. For
> SGLang, it uses them for `--dist-init-addr` and `--node-rank`. Custom launch commands can reference
> the aliases when they need the same topology values.

The immutable operator origin version controls this behavior. An operator-only upgrade does not add
the variables or change the command line of an existing DGD, so it does not roll that workload. To
use the aliases, create the DGD with Dynamo Operator 1.6.0 or later. When Grove is managed outside
the Dynamo platform chart, install Grove `v0.1.0-alpha.14-rc1` or later before creating the DGD.
