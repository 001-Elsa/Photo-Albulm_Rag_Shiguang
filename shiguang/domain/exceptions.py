class DomainError(Exception):
    """可安全映射为业务错误的基类。"""


class ConflictError(DomainError):
    pass


class NotFoundError(DomainError):
    pass


class AuthenticationError(DomainError):
    pass


class AuthorizationError(DomainError):
    pass


class StaleJobError(DomainError):
    pass
