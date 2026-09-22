from app.cases.smart_route.schemas import RoutingRequest, RoutingResponse
from app.cases.smart_route.solver import solve


def plan_routes(request: RoutingRequest) -> RoutingResponse:
    return solve(request)
